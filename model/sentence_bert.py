from locale import str
from multiprocessing import pool
import numpy as np
from nltk import sent_tokenize
np.random.seed(2019)
from numpy import genfromtxt

import random as r
r.seed(2019)
import sys
from sklearn.metrics import fbeta_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
#from model.CorNet import *
import os
os.environ['PYTHONHASHSEED'] = str(2019)
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from query_list import *

import pandas as pd
from argparse import ArgumentParser

import random, sys, math, gzip
from tqdm import tqdm
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score,classification_report, accuracy_score,multilabel_confusion_matrix
import pickle
des_df=pd.read_csv('technique_description_f.csv')

from model.CorNet import *
from model.encode_document import encode_description_finetuned

#max_len=70#40:70 #80:55

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false
    In place operation
    :param tns:
    :return:
    """

    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval

def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def here(subpath=None):
    """
    :return: the path in which the package resides (the directory containing the 'former' dir)
    """
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', subpath))

def contains_nan(tensor):
    return bool((tensor != tensor).sum() > 0)
class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply averaging hidden states
        mean_tensor = hidden_states.mean(dim=1)
        pooled_output = self.dense(mean_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
class BertModel(nn.Module):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        
        
    """
  
    
    
    def __init__(self, config,des_emb_list=None):
        super(BertModel, self).__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config,des_emb_list)
        self.pooler = BertPooler(config)
 

    def forward(self, inputs_embeds, attention_mask=None, position_ids=None,des_tech=None):
        """ Forward pass on the Model.
        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
        Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    
        
        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762
        """
        
        input_shape = inputs_embeds.size()[:-1]
        #des_tech=des_tech.expand(inputs_embeds.size()[0],des_tech.size()[0],des_tech.size()[1])
        #tech_shape=des_tech.size()[:-1]
        device = inputs_embeds.device
        
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
            #new_attention_mask=torch.ones(tech_shape, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
            #extended_new_attention_mask=new_attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        
            extended_attention_mask = attention_mask[:, None, None, :]
            #extended_new_attention_mask=new_attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape, attention_mask.shape))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
       
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        #extended_new_attention_mask = extended_new_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        #extended_new_attention_mask = (1.0 - extended_new_attention_mask) * -10000.0
        #print(np.shape(extended_attention_mask))

        embedding_output = self.embeddings(inputs_embeds=inputs_embeds, position_ids=position_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        #print("aaa",pooled_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
class BertEmbeddings(nn.Module):
    """input sentence embeddings inferred by bottom pre-trained BERT, contruct position embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        
        self.config = config
        self.position_embeddings = nn.Embedding(config.seq_length, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs_embeds, position_ids=None):
    
        input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        device = inputs_embeds.device
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)


        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = inputs_embeds + position_embeddings 
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
class BertSelfAttention(nn.Module):
    
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.config = config
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size/config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        # INPUT:  x = [batch size, sequence length, hidden size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)# [batch size, num of head, sequence length, intermediate size]

    def forward(self, hidden_states, attention_mask=None):
        
        # hidden state:[batch size, sequence length, intermediate size]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        #print(np.shape(query_layer),np.shape(key_layer.transpose(-1, -2)))
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        del context_layer,attention_probs
        return outputs
class BertSelfAttention_tech(nn.Module):
    
    def __init__(self, config,des_emb_list=None):
        super(BertSelfAttention_tech, self).__init__()
        self.config = config
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size/config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.tech_description_original=des_emb_list
        self.tech_description =des_emb_list
        
    def transpose_for_scores(self, x):
        # INPUT:  x = [batch size, sequence length, hidden size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)# [batch size, num of head, sequence length, intermediate size]

    def forward(self, hidden_states ,attention_mask=None):
        
        tech_description=self.tech_description.expand(np.shape(hidden_states)[0],np.shape(self.tech_description_original)[0],np.shape(self.tech_description_original)[1])
        # hidden state:[batch size, sequence length, intermediate size]

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(tech_description)
        mixed_value_layer = self.value(tech_description)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        #print(np.shape(query_layer),np.shape(key_layer.transpose(-1, -2)))
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        #print(attention_mask)
        
        #if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            #attention_scores = attention_scores + attention_mask
        #print(np.shape(query_layer))
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        del context_layer,attention_probs
        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
class BertfirstAttention(nn.Module):
    def __init__(self,config,des_emb_list=None):
        super(BertfirstAttention, self).__init__()
        self.config = config
        self.self = BertSelfAttention_tech(config,des_emb_list)
        self.self2= BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        

    def forward(self, hidden_states, attention_mask=None,tech_mask=None):
        
        self_outputs = self.self(hidden_states, attention_mask=attention_mask)
        
        self2_outputs=self.self2(hidden_states, attention_mask=attention_mask)
        
        attention_output = self.output(self_outputs[0], hidden_states)
        attention_output2=self.output(self2_outputs[0], hidden_states)
        #attention_output=torch.cat((attention_output,attention_output2),1)
        #attention_output=self_outputs[0]
        #attention_output=torch.add(attention_output,attention_output2)
       # attention_output=torch.div(attention_output,2)
        attention_output=torch.max(attention_output,attention_output2)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        del attention_output, self_outputs, self2_outputs
        
        return outputs
class BertAttention(nn.Module):
    def __init__(self,config):
        super(BertAttention, self).__init__()
        self.config = config
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        

    def forward(self, hidden_states, attention_mask=None):
 
        self_outputs = self.self(hidden_states, attention_mask)
  
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        del attention_output, self_outputs
        return outputs
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        
        self.intermediate_act_fn = torch.nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
## Will enumerate this in BERTEncoder
class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights


        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        del layer_output
        return outputs
class BertfirstLayer(nn.Module):
    def __init__(self, config,des_emb_list=None):
        super(BertfirstLayer, self).__init__()
        self.config = config
        self.attention = BertfirstAttention(config,des_emb_list)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None,tech_mask=None):
        
        self_attention_outputs = self.attention(hidden_states, attention_mask)

        
        attention_output = self_attention_outputs[0]
        #attention_output=torch.cat((orig_attention_ouput,attention_output),1)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights


        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        #print(np.shape(outputs))
        outputs = (layer_output,) + outputs
        #print(np.shape(outputs[0]))
        del layer_output
        return outputs
class BertEncoder(nn.Module):
    def __init__(self, config,des_emb_list=None):
        super(BertEncoder, self).__init__()
        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        #self.first=BertfirstLayer(config)
        
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None,tech_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        

        for i, layer_module in enumerate(self.layer): ## enumerate layer of bert
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]
            
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            
        '''
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_outputs = self.first(hidden_states,attention_mask, tech_mask)
        hidden_states = layer_outputs[0]

        if self.output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)
        '''
        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
            
        #del all_attentions
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
class BertConfig():
    """
        :class:`~transformers.BertConfig` is the configuration class to store the configuration of a
        `BertModel`.
        Arguments:
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            seq_length: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """


    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=16,
                 num_attention_heads=2,
                 intermediate_size=192,
                 hidden_act="relu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 seq_length=128,
                 initializer_range=0.02,
                 layer_norm_eps=1e-8,
                 output_attentions=True,
                 output_hidden_states=False,
                 num_labels=2):

  
  

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.seq_length = seq_length
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.num_labels = num_labels

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)
def import_data_sen_in_doc(max_len,threshold:str):
    with open('doc_in_sen_emb_'+threshold,'rb') as f:
        train=pickle.load(f)
    f.close()

    train_ = []
    for idx,doc in enumerate(train):
        temp = np.zeros((len(doc),max_len,768)) 
        print(len(doc))
        for idxidx,sen_itr in enumerate(doc):
            if sen_itr.shape[0]<=max_len:
                temp[idxidx][:sen_itr.shape[0],:]=sen_itr
            else:
                temp[idxidx][:max_len,:] = sen_itr[:max_len,:]
        print(np.shape(temp))
        train_.append(temp)

    
    return np.array(train_,dtype=object)
def import_data_sen(max_len,threshold:str):



    with open("sentence_encode_train","rb") as f:
        train=pickle.load(f)
    with open("sentence_encode_valid","rb") as f:
        valid=pickle.load(f)
    with open("sentence_encode_test","rb") as f:
        test=pickle.load(f)

    f.close()

    #pre_trained_train = [train[i] for i in train]
    #pre_trained_test = [test[i] for i in test]

    #pre_trained = pre_trained_train+pre_trained_test
    # Padding
    
    # assert np.array_equal(neg[0],pre_trained_neg[0][:max_len,:])
    train_ = np.zeros((len(train),max_len,768)) 
    valid_=np.zeros((len(valid),max_len,768)) 
    test_ = np.zeros((len(test),max_len,768))
    # assigning values
    for idx,doc in enumerate(train):
        doc=np.array(doc) 
        if doc.shape[0]<=max_len:
            train_[idx][:doc.shape[0],:] = doc
        else:
            train_[idx][:max_len,:] = doc[:max_len,:]
            
    print('positive example shape: ', train_.shape)
    
    # assert np.array_equal(pos[0],pre_trained_pos[0][:max_len,:])



    for idx,doc in enumerate(valid): 
        doc=np.array(doc) 
        if doc.shape[0]<=max_len:
            valid_ [idx][:doc.shape[0],:] = doc
        else:
            valid_ [idx][:max_len,:] = doc[:max_len,:]
    for idx,doc in enumerate(test): 
        doc=np.array(doc) 
        if doc.shape[0]<=max_len:
            test_ [idx][:doc.shape[0],:] = doc
        else:
            test_ [idx][:max_len,:] = doc[:max_len,:]

    return np.array(train_),np.array(valid_),np.array(test_)
def import_data_f(max_len,threshold:str):



    with open("doc_embedding_"+threshold+"_top5","rb") as f:
        train=pickle.load(f)

    with open("doc_embedding_"+threshold+"_top5_test","rb") as f:
        test=pickle.load(f)

    f.close()

    #pre_trained_train = [train[i] for i in train]
    #pre_trained_test = [test[i] for i in test]

    #pre_trained = pre_trained_train+pre_trained_test
    # Padding
    
    train_ = np.zeros((len(train),max_len,896)) 
    test_ = np.zeros((len(test),max_len,896))
    # assigning values
    for idx,doc in enumerate(train):
        doc=np.array(doc) 
        #print(type(doc))
        if doc.shape[0]<=max_len:
            train_[idx][:doc.shape[0],:] = doc
        else:
            train_[idx][:max_len,:] = doc[:max_len,:]
            
    print('positive example shape: ', train_.shape)
    
    # assert np.array_equal(pos[0],pre_trained_pos[0][:max_len,:])




    for idx,doc in enumerate(test): 
        doc=np.array(doc) 
        if doc.shape[0]<=max_len:
            test_ [idx][:doc.shape[0],:] = doc
        else:
            test_ [idx][:max_len,:] = doc[:max_len,:]
            
    print('negative example shape: ', test_ .shape)
    print('negative example size', sizeof_fmt(sys.getsizeof(test_ )))
    # assert np.array_equal(neg[0],pre_trained_neg[0][:max_len,:])


    return train_,test_
def import_data(max_len,threshold:str):



    with open("sentence_encode_train_"+threshold,"rb") as f:
        train=pickle.load(f)

    with open("sentence_encode_test_"+threshold,"rb") as f:
        test=pickle.load(f)

    f.close()

    #pre_trained_train = [train[i] for i in train]
    #pre_trained_test = [test[i] for i in test]

    #pre_trained = pre_trained_train+pre_trained_test
    # Padding
    
    train_ = np.zeros((len(train),max_len,768)) 
    test_ = np.zeros((len(test),max_len,768))
    # assigning values
    for idx,doc in enumerate(train):
        doc=np.array(doc) 
        #print(type(doc))
        if doc.shape[0]<=max_len:
            train_[idx][:doc.shape[0],:] = doc
        else:
            train_[idx][:max_len,:] = doc[:max_len,:]
            
    print('positive example shape: ', train_.shape)
    
    # assert np.array_equal(pos[0],pre_trained_pos[0][:max_len,:])




    for idx,doc in enumerate(test): 
        doc=np.array(doc) 
        if doc.shape[0]<=max_len:
            test_ [idx][:doc.shape[0],:] = doc
        else:
            test_ [idx][:max_len,:] = doc[:max_len,:]
            
    print('negative example shape: ', test_ .shape)
    print('negative example size', sizeof_fmt(sys.getsizeof(test_ )))
    # assert np.array_equal(neg[0],pre_trained_neg[0][:max_len,:])


    return train_,test_
class PlainC(nn.Module):
    def __init__(self, labels_num, hidden_size, n_probes):
        super(PlainC, self).__init__()
        self.out_mesh_dstrbtn = nn.Linear(hidden_size * n_probes, labels_num)
        nn.init.xavier_uniform_(self.out_mesh_dstrbtn.weight)

    def forward(self, context_vectors):
        output_dstrbtn = self.out_mesh_dstrbtn(context_vectors)  
        return output_dstrbtn
class HTransformer(nn.Module):
    """
    Sentence-level transformer, several transformer blokcs + softmax layer
    
    """

    def __init__(self, config,des_emb_list):
        """
        :param emb_size: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_classes: Number of classes.
     
        """
        super(HTransformer,self).__init__()
        
        self.config = config
        self.des_emb_list=des_emb_list
        self.num_labels = config.num_labels
        #self.multihead_atten=nn.MultiheadAttention(config.hidden_size, 12,batch_first=True)
        self.bert = BertModel(config,des_emb_list)
        #self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Sequential(
            #nn.BatchNorm1d(config.hidden_size),
            
          #  nn.Linear(config.hidden_size, 330),
            #nn.Dropout(0.1),
            nn.Linear(config.hidden_size, self.config.num_labels),
            nn.Dropout(0.15),
            #nn.Linear(config.seq_length*config.hidden_size, self.config.num_labels),
            nn.Tanh()
        )
        #nn.init.xavier_uniform_(self.classifier)
        self.plaincls=PlainC(self.config.num_labels, config.hidden_size, n_probes=1)
        self.cornet = CorNet(self.config.num_labels)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        outputs = self.bert(attention_mask=None,
                            position_ids=None,
                            inputs_embeds=x,des_tech=self.des_emb_list)
        
        pooled_output = outputs[1]

        logits = self.classifier(pooled_output)

        #logits = self.plaincls(pooled_output)
        logits=self.cornet(logits)
        outputs = (logits,) + outputs[2:]
        
        del logits
        return outputs
class relevant_classifier(nn.Module):
    """
    Sentence-level transformer, several transformer blokcs + softmax layer
    
    """
    def __init__(self, config,des_emb_list):
        """
        :param emb_size: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_classes: Number of classes.
        
        """
        super(HTransformer2,self).__init__()
        
        self.config = config
        self.des_emb_list=des_emb_list
        self.num_labels = config.num_labels
        self.multihead_atten=nn.MultiheadAttention(config.hidden_size, 16,batch_first=True)
        self.bert = BertModel(config,des_emb_list)
        #self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Sequential(
            #nn.LayerNorm(config.hidden_size),
            
            #nn.Linear(config.hidden_size, 330),

            nn.Linear(config.hidden_size, self.config.num_labels),
            nn.Dropout(0.15),
            #nn.Linear(config.seq_length*config.hidden_size, self.config.num_labels),
            nn.Tanh()
        )
        
        #self.plaincls=PlainC(self.config.num_labels, config.hidden_size, n_probes=1)
        self.cornet = CorNet(self.config.num_labels)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        outputs = self.bert(attention_mask=None,
                            position_ids=None,
                            inputs_embeds=x)

        #pooled_output = outputs[1]
        #print("hhhh",np.shape(outputs[0][0])) (batch size * number of sent * 768)
        #print(self.des_emb_list)
        des_emb_list=self.des_emb_list.expand(np.shape(outputs[0])[0],np.shape(self.des_emb_list)[0],np.shape(self.des_emb_list)[1])
        #print(np.shape(des_emb_list),np.shape(outputs[0]))
        concat=torch.cat((outputs[0],des_emb_list),1)
        attn_output,attn_output_weight=self.multihead_atten(concat,concat,concat)

        attn_output=torch.mean(attn_output,1)


        #logits = self.classifier(attn_output)
        logits=self.classifier(attn_output)
        logits=self.cornet(logits)
        outputs = (logits,) + outputs[2:]
        
        del logits
        return outputs
'''
 * HTransformer2()-Sentence-level transformer, several transformer blokcs 
'''
class HTransformer2(nn.Module):
    '''
    * __init__()-Initial model 
    * @config: BERT config
    * @des_emb_list: list of technique description embeddings
    '''
    def __init__(self, config,des_emb_list):
        super(HTransformer2,self).__init__()

        np.random.seed(42)
        self.config = config
        self.des_emb_list=des_emb_list
        self.num_labels = config.num_labels
        self.multihead_atten=nn.MultiheadAttention(config.hidden_size, 16,batch_first=True)
        self.bert = BertModel(config,des_emb_list)
        #self.dropout = nn.Dropout(0.15)

        self.classifier = nn.Sequential(

            nn.Linear(config.hidden_size, self.config.num_labels),
            nn.Dropout(0.15)
        )
        
        self.cornet = CorNet(self.config.num_labels)
        self.Tan=nn.Tanh()
    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        outputs = self.bert(attention_mask=None,
                            position_ids=None,
                            inputs_embeds=x)

        #pooled_output = outputs[1]
        #print("hhhh",np.shape(outputs[0][0])) (batch size * number of sent * 768)
        #print(self.des_emb_list)
        des_emb_list=self.des_emb_list.expand(np.shape(outputs[0])[0],np.shape(self.des_emb_list)[0],np.shape(self.des_emb_list)[1])
        #print(np.shape(des_emb_list),np.shape(outputs[0]))
        concat=torch.cat((outputs[0],des_emb_list),1)
        attn_output,attn_output_weight=self.multihead_atten(concat,concat,concat)
       
        attn_output=attn_output[:,:np.shape(outputs[0])[1],:]
        #print(attn_output.shape)
        attn_output=torch.max(attn_output,1)
        '''
        for idx,att_iter in enumerate(attn_output_weight):
            weight_of_tech=[0]*np.shape(self.des_emb_list)[0]
            for idxidx, atten_iter in enumerate(att_iter):
                for idxidxidx, attention_iter in enumerate(atten_iter):
                    if(idxidxidx>=np.shape(outputs[0])[1]):
                        weight_of_tech[idxidxidx-np.shape(outputs[0])[1]]=weight_of_tech[idxidxidx-np.shape(outputs[0])[1]]+atten_iter[idxidxidx]
            print(weight_of_tech)
        '''
        #logits = self.classifier(attn_output)

        logits=self.classifier(attn_output[0])
        logits=self.cornet(logits)
        #logits=self.dropout(logits)
        logits=self.Tan(logits)
        outputs = (logits,) + outputs[2:]
        
        del logits
        return outputs
'''
 * HTransformer2()-Sentence-level transformer, several transformer blokcs 
'''
class HTransformer3(nn.Module):
    '''
    * __init__()-Initial model 
    * @config: BERT config
    * @des_emb_list: list of technique description embeddings
    '''
    def __init__(self, config,des_emb_list):
        super(HTransformer3,self).__init__()

        np.random.seed(42)
        self.config = config
        self.des_emb_list=des_emb_list
        self.num_labels = config.num_labels
        self.multihead_atten=nn.MultiheadAttention(config.hidden_size, 16,batch_first=True)
        self.bert = BertModel(config,des_emb_list)
        #self.dropout = nn.Dropout(0.15)

        self.classifier = nn.Sequential(

            nn.Linear(config.hidden_size, self.config.num_labels),
            nn.Dropout(0.15)
        )
        
        self.cornet = CorNet(self.config.num_labels)
        self.Tan=nn.Tanh()
    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        outputs = self.bert(attention_mask=None,
                            position_ids=None,
                            inputs_embeds=x)

        #pooled_output = outputs[1]
        #print("hhhh",np.shape(outputs[0][0])) (batch size * number of sent * 768)
        #print(self.des_emb_list)
        des_emb_list=self.des_emb_list.expand(np.shape(outputs[0])[0],np.shape(self.des_emb_list)[0],np.shape(self.des_emb_list)[1])
        #print(np.shape(des_emb_list),np.shape(outputs[0]))
        concat=torch.cat((outputs[0],des_emb_list),1)
        attn_output,attn_output_weight=self.multihead_atten(concat,concat,concat)
        attn_output=attn_output[:,:np.shape(outputs[0])[1],:]
        attn_output=torch.max(attn_output,1)
        

        #logits = self.classifier(attn_output)
        logits=self.classifier(attn_output[0])
        logits=self.cornet(logits)
        #logits=self.dropout(logits)
        logits=self.Tan(logits)
        outputs = (logits,) + outputs[2:]
        
        del logits
        return outputs



random.seed(0)
class MITREtrievalmodel(nn.Module):
    """
    Sentence-level transformer, several transformer blokcs + softmax layer
    
    """
    def __init__(self, config,des_emb_list):
        """
        :param emb_size: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_classes: Number of classes.
        
        """
        super(MITREtrievalmodel,self).__init__()
        self.config = config
        
        #torch.manual_seed(0)
        np.random.seed(0)
        self.des_emb_list=des_emb_list
        self.num_labels = config.num_labels
        self.multihead_atten=nn.MultiheadAttention(config.hidden_size, 16,batch_first=True)
        self.bert = BertModel(config,des_emb_list)
        
        self.classifier = nn.Sequential(
            #nn.LayerNorm(config.hidden_size),
            
            #nn.Linear(config.hidden_size, 330),
            #nn.Dropout(0.15),
            nn.Linear(config.hidden_size, self.config.num_labels),
            nn.Dropout(0.15)
            #nn.Linear(config.seq_length*config.hidden_size, self.config.num_labels),

        )
        
        self.l1_nn=nn.Linear(self.config.num_labels,self.config.num_labels)
        self.l2_nn=nn.Linear(self.config.num_labels,self.config.num_labels)
        self.l3_nn=nn.Linear(self.config.num_labels,self.config.num_labels)
        self.must_nn=nn.Linear(self.config.num_labels,self.config.num_labels)
        self.cornet = CorNet(self.config.num_labels)
        #self.dropout = nn.Dropout(0.15)
        self.tan=nn.Tanh()
        #self.final_output=nn.Linear(self.config.num_labels,self.config.num_labels)
    def forward(self, x,ontology_answer_l1=None,ontology_answer_l2=None,ontology_answer_l3=None, must_ttp_ans=None,must_not_ttp_ans=None):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        outputs = self.bert(attention_mask=None,
                            position_ids=None,
                            inputs_embeds=x)

        #pooled_output = outputs[1]
        #print("hhhh",np.shape(outputs[0][0])) (batch size * number of sent * 768)
        #print(self.des_emb_list)
        des_emb_list=self.des_emb_list.expand(np.shape(outputs[0])[0],np.shape(self.des_emb_list)[0],np.shape(self.des_emb_list)[1])
        #print(np.shape(des_emb_list),np.shape(outputs[0]))
        concat=torch.cat((outputs[0],des_emb_list),1)
        attn_output,attn_output_weight=self.multihead_atten(concat,concat,concat)
        attn_output=attn_output[:,:np.shape(outputs[0])[1],:]
        attn_output=torch.mean(attn_output,1)


        logits = self.classifier(attn_output)
        #logits=self.classifier(outputs[1])
        logits=self.tan(logits)
        logits=self.cornet(logits)
        
        #whole_ans=torch.cat((logits,ontology_answer_l1),1)
        #+++++++++++++++++++++++++++++++++++++++
        #ontology_answer_l1=self.tan(self.l1_nn(ontology_answer_l1))
        #ontology_answer_l2=self.tan(self.l1_nn(ontology_answer_l2))
        #ontology_answer_l3=self.tan(self.l1_nn(ontology_answer_l3))
        #must_ttp_ans=self.tan(self.must_nn(must_ttp_ans))
        for r in range(0,logits.shape[0]):
            for c in range(0, logits.shape[1]):
                    if logits[r][c] > 0 and ontology_answer_l1[r][c]==1:
                            logits[r][c] = (logits[r][c]+1)/2
                    if ontology_answer_l2[r][c]==1 or ontology_answer_l3[r][c]==1 or must_ttp_ans[r][c]==1:
                            logits[r][c] = 1
                    if must_not_ttp_ans[r][c]==1:
                            logits[r][c]=0

        
        outputs = (logits,) + outputs[2:]
        
        del logits
        return outputs
BertLayerNorm = torch.nn.LayerNorm
def init_weights(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, BertLayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()  
class Normalize(object):
    
    def normalize(self, X_train, X_val,max_len):
        self.scaler =MinMaxScaler()
        X_train, X_val = X_train.reshape(X_train.shape[0],-1),X_val.reshape(X_val.shape[0],-1)
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        X_train, X_val = X_train.reshape(X_train.shape[0],max_len,-1), X_val.reshape(X_val.shape[0],max_len,-1)
       
        return (X_train, X_val) 
    
    def inverse(self, X_train, X_val):
        X_train = self.scaler.inverse_transform(X_train)
        X_val   = self.scaler.inverse_transform(X_val)
    
        return (X_train, X_val) 


def import_ontology_answer(threshold:str,Technique_name:list,train=''):
    f=open("all_answer_from_ontology_"+threshold+train,"rb")          
    ontology_feature=pickle.load(f)
    print(np.shape(ontology_feature))

    ontology_answer_l1=torch.zeros((np.shape(ontology_feature)[0], len(Technique_name)))
    ontology_answer_l2=torch.zeros((np.shape(ontology_feature)[0], len(Technique_name)))
    ontology_answer_l3=torch.zeros((np.shape(ontology_feature)[0], len(Technique_name)))
    must_ttp_ans=torch.zeros((np.shape(ontology_feature)[0], len(Technique_name)))
    must_not_ttp_ans=torch.zeros((np.shape(ontology_feature)[0], len(Technique_name)))
    for indexes,ans in enumerate(ontology_feature):
        l1_ans_cor=ans[0]+ans[1]+ans[2]
        l2_ans_cor=ans[3]+ans[4]
        l3_ans_cor=ans[5]
        must_ttp=ans[6]
        must_not_ttp=ans[7]
        temp1=torch.zeros(( len(Technique_name)))
        temp2=torch.zeros(( len(Technique_name)))
        temp3=torch.zeros(( len(Technique_name)))
        temp4=torch.zeros(( len(Technique_name)))
        temp5=torch.zeros(( len(Technique_name)))
        for l1_ans in l1_ans_cor: 
                if l1_ans.upper() in Technique_name:
                        temp1[Technique_name.index(l1_ans.upper())]=1
        for l2_ans in l2_ans_cor:
                if l2_ans.upper() in Technique_name:
                        #print(l2_ans.upper(),Technique_name.index(l2_ans.upper()))
                        temp2[Technique_name.index(l2_ans.upper())]=1
        for l3_ans in l3_ans_cor :
                if l3_ans.upper() in Technique_name:
                        #print(l3_ans.upper(),Technique_name.index(l3_ans.upper()))
                        temp3[Technique_name.index(l3_ans.upper())]=1
        for l4_ans in must_ttp :
                if l4_ans.upper() in Technique_name:
                        #print("must=",l4_ans.upper(),Technique_name.index(l4_ans.upper()))
                        temp4[Technique_name.index(l4_ans.upper())]=1
        for l5_ans in must_not_ttp :
                if l5_ans.upper() in Technique_name:
                        # print("must=",l5_ans.upper(),Technique_name.index(l4_ans.upper()))
                        temp5[Technique_name.index(l5_ans.upper())]=1
        #print(temp2)
        ontology_answer_l1[indexes]=temp1
        ontology_answer_l2[indexes]=temp2
        ontology_answer_l3[indexes]=temp3
        must_ttp_ans[indexes]=temp4
        must_not_ttp_ans[indexes]=temp5

    print(np.shape(ontology_answer_l1),np.shape(ontology_answer_l2))
    return ontology_answer_l1,ontology_answer_l2,ontology_answer_l3,must_ttp_ans,must_not_ttp_ans
'''
 * MITREtrieval()- Model Training and testing
 * @threshold: Technique Threshold
'''
def MITREtrieval(threshold:str):
    Technique_name_f=open("Data/Technique_name_"+threshold+".txt","rb")
    Technique_name=pickle.load(Technique_name_f)
    Technique_name_f.close()
    config = BertConfig(num_labels=len(Technique_name), num_attention_heads=12)
    ontology_answer_l1,ontology_answer_l2,ontology_answer_l3,must_ttp_ans,must_not_ttp_ans=import_ontology_answer(threshold,Technique_name,'train')
    ontology_answer_l1_t,ontology_answer_l2_t,ontology_answer_l3_t,must_ttp_ans_t,must_not_ttp_ans_t=import_ontology_answer(threshold,Technique_name)
    max_len=125
    #max_len=65#40:70 #80:55
    lr=2e-5
    EPOCH=301
    gradient_clipping = 1.0
    train_batch=16
    val_batch=1
    train_emb,test_emb=import_data(max_len,threshold)
    #print(np.shape(df_procedure))
    des_emb_list=encode_description_finetuned(des_df,Technique_name)
    des_emb_list=torch.FloatTensor(des_emb_list).cuda()
    model = MITREtrievalmodel(config,des_emb_list)

    model.apply(init_weights)
    #model.to('cuda:1')
    model.cuda()
    #if torch.cuda.device_count() > 1:
        #model= torch.nn.DataParallel(model)
    
    opt = torch.optim.Adam(lr=lr, params=model.parameters())
    losses = []
    macro_f = []
    
    with open('dl_train_feature_'+threshold+'_clean',"rb") as f:
            train_=pickle.load(f)
    with open("dl_test_feature_"+threshold+"_clean","rb") as of:
            test_=pickle.load(of)
    f.close()
    of.close()
    x_train=train_emb
    y_train=train_['list']
    x_test=test_emb
    y_test=test_['list']
    normalizer = Normalize()
    print("su ",np.shape(x_train),np.shape(y_train))
    x_train, x_test = normalizer.normalize(x_train, x_test,max_len) 
    print(y_train,y_test)
    tensor_train_x = torch.from_numpy(x_train).type(torch.FloatTensor)
    tensor_train_y = torch.FloatTensor(y_train).type(torch.FloatTensor)

    tensor_val_x = torch.from_numpy(x_test).type(torch.FloatTensor)
    tensor_val_y = torch.FloatTensor(y_test).type(torch.FloatTensor)

    training_set = torch.utils.data.TensorDataset(tensor_train_x,tensor_train_y, ontology_answer_l1,ontology_answer_l2,ontology_answer_l3,must_ttp_ans,must_not_ttp_ans)# create your datset
    val_set = torch.utils.data.TensorDataset(tensor_val_x,tensor_val_y,ontology_answer_l1_t,ontology_answer_l2_t,ontology_answer_l3_t,must_ttp_ans_t,must_not_ttp_ans_t)

    
    trainloader=torch.utils.data.DataLoader(training_set, batch_size=train_batch, shuffle=True, num_workers=1)
    testloader=torch.utils.data.DataLoader(val_set, batch_size=val_batch, shuffle=False, num_workers=1)
    loss_weight = ((tensor_train_y.shape[0] / torch.sum(tensor_train_y, dim=0))-1).cuda()
    print(loss_weight)
    #del training_set, val_set
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weight)
    #loss=f2_loss()
    max_f1=0
    #loss_list=[]
    for e in tqdm(range(EPOCH)):
        print('\n epoch ',e)
        epoch_loss = 0.0
        for i, data in enumerate(tqdm(trainloader)):
            model.train(True)
            
            inputs, labels, l1,l2,l3,must,must_not = data
            if inputs.size(1) > config.seq_length:
                inputs = inputs[:, :config.seq_length, :]

            if torch.cuda.is_available():
                # inputs, labels = Variable(inputs.to('cuda')), labels.to('cuda')
                inputs, labels = Variable(inputs.cuda()), labels.cuda()
            #print(type(inputs))
            out = model(inputs,l1.cuda(),l2.cuda(),l3.cuda(),must.cuda(),must_not.cuda())
            #print(out[0])
            #out[0]=out[0].requires_grad_()
            output = loss(out[0], labels)
            epoch_loss += (output.item())
            
            output.backward()
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            del inputs, labels, out, output
            torch.cuda.empty_cache()


            #losses.append(train_loss_tol)
            opt.zero_grad()
        #print("n",train_batch,"s",tensor_val_y.shape[0])
        epoch_loss /= int(tensor_train_y.shape[0] / train_batch)  # divide by number of batches per epoch 
        losses.append(epoch_loss)
    
        print("EPOCH_LOSS= ",epoch_loss)
        if e % 5==0:
            with torch.no_grad():
                model.train(False)
                predictions = torch.empty((tensor_val_y.shape[0], len(Technique_name)))
                y_eval_pred = []
                y_eval_true = []
                y_eval_prob = []

                y_train_pred = []
                y_train_true = []
                y_train_prob = []

                train_attention_scores = torch.Tensor()
                attention_scores = torch.Tensor()
                for idx, data in enumerate(tqdm(testloader)):
                    inputs, labels, l1,l2,l3,must,must_not  = data
                    if inputs.size(1) > config.seq_length:
                        inputs = inputs[:, :config.seq_length, :]
                    if torch.cuda.is_available():
                        inputs, labels = Variable(inputs.cuda()), labels.cuda()
                    out = model(inputs,l1.cuda(),l2.cuda(),l3.cuda(),must.cuda(),must_not.cuda())
                    pred_prob = out[0].cpu()
                    predictions[idx]=pred_prob
                    '''
                    if config.output_attentions:
                        last_layer_attention = out[1][-1].cpu()
                        attention_scores = torch.cat((attention_scores, last_layer_attention))
        #                         attention_scores=attention_scores+[last_layer_attention]
                    '''
                    del inputs, labels, out

                
                for r in range(0, predictions.shape[0]):
                    for c in range(0, predictions.shape[1]):
                        if predictions[r][c] > 0:
                            predictions[r][c] = 1
                        else:
                            predictions[r][c] = 0
                
                #print(predictions,predictions.shape)
                precision=precision_score(y_true =tensor_val_y.cpu(), y_pred = predictions.cpu(),average='micro')
                recall=recall_score(y_true =tensor_val_y.cpu(), y_pred = predictions.cpu(),average='micro')
                f1=fbeta_score(y_true =tensor_val_y.cpu(), y_pred = predictions.cpu(), average='micro', beta=2)

                f1m=fbeta_score(y_true =tensor_val_y.t().cpu(), y_pred = predictions.t().cpu(), average='macro', beta=2)
                print("F2 score= ",f1,f1m)
  
                
                if(f1>max_f1):
                    fpr=1-recall
                    print("MAX F1 score= ",f1," macro= ",f1m," fpr= ",fpr)
                    max_f1=f1
                    confusion=pd.DataFrame(multilabel_confusion_matrix(y_true =tensor_val_y.cpu(), y_pred = predictions.cpu()).reshape(np.shape(predictions)[1],2*2))
                    confusion.to_csv('MITretrieval_'+threshold+'.csv',index=True)
                    with open("ans_"+threshold,"wb") as cor:
                        pickle.dump(predictions,cor)
                    cm=multilabel_confusion_matrix(y_true =tensor_val_y.cpu(), y_pred = predictions.cpu()).reshape(np.shape(predictions)[1],2*2)
                    #print(cm,np.shape(cm))
                    tn=0
                    fp=0
                    fn=0
                    tp=0
                    fpr=[]
                    fnr=[]
                    for rr in cm:
                        tn=tn+rr[0]
                        fp=fp+rr[1]
                        fn=fn+rr[2]
                        tp=tp+rr[3]
                        fpr.append(rr[1]/(rr[0]+rr[1]))
                        fnr.append(rr[2]/(rr[2]+rr[3]))
                    print(fp,fp/(tn+fp),fn/(tp+fn),sum(fpr)/len(fpr),sum(fnr)/len(fnr))
                    torch.save(model.state_dict(),"sentence_BERT_model_"+threshold+".pt")
                print("Current max F1 score= ",max_f1)
                clsf_report = pd.DataFrame(classification_report(y_true =tensor_val_y.t().cpu(), y_pred = predictions.t().cpu(), output_dict=True)).transpose()
                clsf_report.to_csv('newbert_'+str(e)+'report_result2.csv', index= True) 
                clsf_report = pd.DataFrame(classification_report(y_true =tensor_val_y.cpu(), y_pred = predictions.cpu(), output_dict=True,target_names=Technique_name)).transpose()
                clsf_report.to_csv('newbert_'+str(e)+'result2.csv', index= True)
    with open("loss_file_sen"+str(tensor_train_y.shape[0]),"wb") as fff:
                pickle.dump(losses,fff)
    fff.close()
def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets.squeeze())

def one_sentence_model_train(threshold:str):

    Technique_name_f=open("Data/Technique_name_sentence.txt","rb")
    Technique_name=pickle.load(Technique_name_f)
    Technique_name_f.close()
    config = BertConfig(num_labels=len(Technique_name))
    print(Technique_name)
    max_len=50
    #max_len=65#40:70 #80:55
    lr=2e-5
    EPOCH=101
    gradient_clipping = 1.0
    train_batch=16
    val_batch=1
    train_emb,valid_emb,test_emb=import_data_sen(max_len,threshold)
    #print(np.shape(df_procedure))
    des_emb_list=encode_description_finetuned(des_df,Technique_name)
    des_emb_list=torch.FloatTensor(des_emb_list).cuda()

    model = HTransformer2(config,des_emb_list)
    model.apply(init_weights)
    #model.to('cuda:1')
    model.cuda()
    #if torch.cuda.device_count() > 1:
        #model= torch.nn.DataParallel(model)
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    opt = torch.optim.Adam(lr=lr, params=model.parameters())
    #opt=torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    losses = []
    macro_f = []
    
    with open('sentence_train',"rb") as f:
            train_=pickle.load(f)
    with open('sentence_valid',"rb") as of:
            valid_=pickle.load(of)
    with open('sentence_test',"rb") as of:
            test_=pickle.load(of)
    f.close()
    of.close()
    

    possible_labels = train_.label.unique()
    possible_labels_val=valid_.label.unique()
    possible_labels_test=test_.label.unique()
    #print(possible_labels)
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    train_['label'] = train_.label.replace(label_dict)
    label_dict_val = {}
    for index, possible_label in enumerate(possible_labels_val):
        label_dict_val[possible_label] = index
    valid_['label'] = valid_.label.replace(label_dict_val)
    label_dict_test = {}
    for index, possible_label in enumerate(possible_labels_test):
        label_dict_test[possible_label] = index
    test_['label'] = test_.label.replace(label_dict_test)
    x_train=train_emb
    y_train=train_['label']
    x_valid=valid_emb
    y_valid=valid_['label']
    x_test=test_emb
    y_test=test_['label']

    #y_train = LabelBinarizer().fit_transform(y_train).tolist()
    #y_valid = LabelBinarizer().fit_transform(y_valid).tolist()
    normalizer = Normalize()
    print("su ",type(x_train),np.shape(x_train),y_train)
    x_train, x_valid = normalizer.normalize(x_train, x_valid,max_len) 
    #print(y_test)
    tensor_train_x = torch.from_numpy(x_train).type(torch.FloatTensor)
    tensor_train_y = torch.FloatTensor(y_train).type(torch.LongTensor)

    tensor_val_x = torch.from_numpy(x_valid).type(torch.FloatTensor)
    tensor_val_y = torch.FloatTensor(y_valid).type(torch.LongTensor)

    training_set = torch.utils.data.TensorDataset(tensor_train_x,tensor_train_y)# create your datset
    val_set = torch.utils.data.TensorDataset(tensor_val_x,tensor_val_y)
    print(tensor_train_y.shape[0] )
    
    trainloader=torch.utils.data.DataLoader(training_set, batch_size=train_batch, shuffle=True, num_workers=1)
    validloader=torch.utils.data.DataLoader(val_set, batch_size=val_batch, shuffle=False, num_workers=1)
    loss_weight = ((tensor_train_y.shape[0] / torch.sum(tensor_train_y, dim=0))-1).cuda()
    print(loss_weight)
    max_f1=0

    for e in tqdm(range(EPOCH)):
        print('\n epoch ',e)
        epoch_loss = 0.0
        for i, data in enumerate(tqdm(trainloader)):
            model.train(True)
            
            inputs, labels = data
            if inputs.size(1) > config.seq_length:
                inputs = inputs[:, :config.seq_length, :]

            if torch.cuda.is_available():
                # inputs, labels = Variable(inputs.to('cuda')), labels.to('cuda')
                inputs, labels = Variable(inputs.cuda()), labels.cuda()
            #print(type(inputs))
            out = model(inputs)
            #print(out[0])
            #out[0]=out[0].requires_grad_()
            #print(labels,np.shape(out[0]))
            output = loss_fn(out[0], labels)
            epoch_loss += (output.item())
            
            output.backward()
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            del inputs, labels, out, output
            torch.cuda.empty_cache()


            #losses.append(train_loss_tol)
            opt.zero_grad()
        #print("n",train_batch,"s",tensor_val_y.shape[0])
        epoch_loss /= int(tensor_train_y.shape[0] / train_batch)  # divide by number of batches per epoch 
        losses.append(epoch_loss)
    
        print("EPOCH_LOSS= ",epoch_loss)
        if e % 5==0:
            with torch.no_grad():
                model.train(False)
                predictions = torch.empty((tensor_val_y.shape[0], len(Technique_name)))
                y_eval_pred = []
                y_eval_true = []
                y_eval_prob = []

                y_train_pred = []
                y_train_true = []
                y_train_prob = []

                train_attention_scores = torch.Tensor()
                attention_scores = torch.Tensor()
                for idx, data in enumerate(tqdm(validloader)):
                    inputs, labels = data
                    if inputs.size(1) > config.seq_length:
                        inputs = inputs[:, :config.seq_length, :]
                    if torch.cuda.is_available():
                        # inputs, labels = Variable(inputs.to('cuda')), labels.to('cuda')
                        inputs, labels = Variable(inputs.cuda()), labels.cuda()

                    out = model(inputs)
                    pred_prob = out[0].cpu()

                    predictions[idx]=pred_prob
                    '''
                    if config.output_attentions:
                        last_layer_attention = out[1][-1].cpu()
                        attention_scores = torch.cat((attention_scores, last_layer_attention))
        #                         attention_scores=attention_scores+[last_layer_attention]
                    '''
                    del inputs, labels, out

                predict=[]
                for r in range(0, predictions.shape[0]):
                    predict.append(torch.argmax(predictions[r]))
                print(predict)
                precision=precision_score(y_true =tensor_val_y.cpu(), y_pred = torch.FloatTensor(predict).cpu(), pos_label=1, average='micro')
                recall=recall_score(y_true =tensor_val_y.cpu(), y_pred =torch.FloatTensor(predict).cpu(), pos_label=0, average='macro')
                f1=fbeta_score(y_true =tensor_val_y.cpu(), y_pred =torch.FloatTensor(predict).cpu(), pos_label=1, average='micro', beta=2)
                f1m=fbeta_score(y_true =tensor_val_y.cpu(), y_pred =torch.FloatTensor(predict).cpu(), average='macro', beta=2)
                print("F1 score= ",f1)
                
                if(f1>max_f1):
                    fpr=1-recall
                    print("MAX F1 score= ",f1,f1m)
                    max_f1=f1
                    cm=multilabel_confusion_matrix(y_true =tensor_val_y.cpu(), y_pred =torch.FloatTensor(predict).cpu()).reshape(34,2*2)
                    tn=0
                    fp=0
                    fn=0
                    tp=0
                    fpr=[]
                    fnr=[]
                    for rr in cm:
                        tn=tn+rr[0]
                        fp=fp+rr[1]
                        fn=fn+rr[2]
                        tp=tp+rr[3]
                        fpr.append(rr[1]/(rr[0]+rr[1]))
                        fnr.append(rr[2]/(rr[2]+rr[3]))
                    print(fp,fp/(tn+fp),fn/(tp+fn),sum(fpr)/len(fpr),sum(fnr)/len(fnr))
                    torch.save(model.state_dict(),"onesentence_BERT_model.pt")
                print("Current max F1 score= ",max_f1)
                clsf_report = pd.DataFrame(classification_report(y_true =tensor_val_y.t().cpu(),  y_pred = torch.FloatTensor(predict).cpu(), output_dict=True)).transpose()
                clsf_report.to_csv('newbert_'+str(e)+'report_result2.csv', index= True) 
                clsf_report = pd.DataFrame(classification_report(y_true =tensor_val_y.cpu(), y_pred = torch.FloatTensor(predict).cpu(), output_dict=True,target_names=possible_labels)).transpose()
                clsf_report.to_csv('newbert_'+str(e)+'result2.csv', index= True)
    with open("loss_file_sen"+str(tensor_train_y.shape[0]),"wb") as fff:
                pickle.dump(losses,fff)
    fff.close()

'''
 * sentence_model_train()-Deep Learning model training without knowledge fusion
 * @threshold: Technique threshold
'''
def sentence_model_train(threshold:str):
    #Load in target technique name
    Technique_name_f=open("Data/Technique_name_"+threshold+".txt","rb")
    Technique_name=pickle.load(Technique_name_f)
    Technique_name_f.close()
    print(np.shape(Technique_name))
    ## Set Config of BERT
    config = BertConfig(num_labels=len(Technique_name), num_attention_heads=8)
    ## Set Config of Deep Neural Network
    max_len=125
    lr=2e-5
    EPOCH=101
    gradient_clipping = 1.0
    train_batch=16
    val_batch=1
    #Import pre-store sentence embedding
    train_emb,test_emb=import_data(max_len,threshold)
    #Import technique description
    des_emb_list=encode_description_finetuned(des_df,Technique_name)
    des_emb_list=torch.FloatTensor(des_emb_list).cuda()

    ## Load model and initilize
    model = HTransformer2(config,des_emb_list)
    model.apply(init_weights)
    model.cuda()
    opt = torch.optim.Adam(lr=lr, params=model.parameters())

    losses = []
    
    ## Load training and testing data
    with open('dl_train_feature_'+threshold+'_clean',"rb") as f:
            train_=pickle.load(f)
    with open("dl_test_feature_"+threshold+"_clean","rb") as of:
            test_=pickle.load(of)
    f.close()
    of.close()
    x_train=train_emb
    y_train=train_['list']
    x_test=test_emb
    y_test=test_['list']
    
    normalizer = Normalize()
    x_train, x_test = normalizer.normalize(x_train, x_test,max_len) 
    tensor_train_x = torch.from_numpy(x_train).type(torch.FloatTensor)
    tensor_train_y = torch.FloatTensor(y_train).type(torch.FloatTensor)
    tensor_val_x = torch.from_numpy(x_test).type(torch.FloatTensor)
    tensor_val_y = torch.FloatTensor(y_test).type(torch.FloatTensor)
    training_set = torch.utils.data.TensorDataset(tensor_train_x,tensor_train_y)# create your datset
    val_set = torch.utils.data.TensorDataset(tensor_val_x,tensor_val_y)
    trainloader=torch.utils.data.DataLoader(training_set, batch_size=train_batch, shuffle=True, num_workers=1)
    testloader=torch.utils.data.DataLoader(val_set, batch_size=val_batch, shuffle=False, num_workers=1)
    loss_weight = ((tensor_train_y.shape[0] / torch.sum(tensor_train_y, dim=0))-1).cuda()

    loss = torch.nn.BCEWithLogitsLoss(pos_weight=loss_weight)
    max_f1=0
    print("===============Start Training=======================")
    for e in tqdm(range(EPOCH)):
        print('\n epoch ',e)
        epoch_loss = 0.0
        for i, data in enumerate(tqdm(trainloader)):
            model.train(True)
            
            inputs, labels = data
            if inputs.size(1) > config.seq_length:
                inputs = inputs[:, :config.seq_length, :]

            if torch.cuda.is_available():
                # inputs, labels = Variable(inputs.to('cuda')), labels.to('cuda')
                inputs, labels = Variable(inputs.cuda()), labels.cuda()
            #print(type(inputs))
            out = model(inputs)
            #print(out[0])
            #out[0]=out[0].requires_grad_()
            output = loss(out[0], labels)
            epoch_loss += (output.item())
            
            output.backward()
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            del inputs, labels, out, output
            torch.cuda.empty_cache()


            #losses.append(train_loss_tol)
            opt.zero_grad()
        #print("n",train_batch,"s",tensor_val_y.shape[0])
        epoch_loss /= int(tensor_train_y.shape[0] / train_batch)  # divide by number of batches per epoch 
        losses.append(epoch_loss)
    
        print("EPOCH_LOSS= ",epoch_loss)
        # Testing every 5 Epoch
        if e % 5==0:
            with torch.no_grad():
                model.train(False)
                predictions = torch.empty((tensor_val_y.shape[0], len(Technique_name)))
                for idx, data in enumerate(tqdm(testloader)):
                    inputs, labels = data
                    if inputs.size(1) > config.seq_length:
                        inputs = inputs[:, :config.seq_length, :]
                    if torch.cuda.is_available():
                        inputs, labels = Variable(inputs.cuda()), labels.cuda()
                    out = model(inputs)
                    pred_prob = out[0].cpu()
                    predictions[idx]=pred_prob
                    del inputs, labels, out

                for r in range(0, predictions.shape[0]):
                    for c in range(0, predictions.shape[1]):
                        if predictions[r][c] > 0:
                            predictions[r][c] = 1
                        else:
                            predictions[r][c] = 0
                
                precision=precision_score(y_true =tensor_val_y.cpu(), y_pred = predictions.cpu(),average='micro')
                recall=recall_score(y_true =tensor_val_y.cpu(), y_pred = predictions.cpu(),pos_label=1,average='macro')
                f1=fbeta_score(y_true =tensor_val_y.cpu(), y_pred = predictions.cpu(), average='micro', beta=2)
                f1m=fbeta_score(y_true =tensor_val_y.cpu(), y_pred = predictions.cpu(), average='macro', beta=2)
                print("F2 score= ",f1)
                
                if(f1>max_f1):
                    fpr=1-recall
                    print("MAX F1 score= ",f1," macro= ",f1m," fpr= ",fpr)
                    max_f1=f1
                    confusion=pd.DataFrame(multilabel_confusion_matrix(y_true =tensor_val_y.cpu(), y_pred = predictions.cpu()).reshape(np.shape(predictions)[1],2*2))
                    confusion.to_csv('MITretrieval_'+threshold+'.csv',index=True)
                    with open("ans_"+threshold,"wb") as cor:
                        pickle.dump(predictions,cor)
                    cm=multilabel_confusion_matrix(y_true =tensor_val_y.cpu(), y_pred = predictions.cpu()).reshape(np.shape(predictions)[1],2*2)
                    #print(cm,np.shape(cm))
                    tn=0
                    fp=0
                    fn=0
                    tp=0
                    fpr=[]
                    fnr=[]
                    for rr in cm:
                        tn=tn+rr[0]
                        fp=fp+rr[1]
                        fn=fn+rr[2]
                        tp=tp+rr[3]
                        fpr.append(rr[1]/(rr[0]+rr[1]))
                        fnr.append(rr[2]/(rr[2]+rr[3]))
                    print(fp,fp/(tn+fp),fn/(tp+fn),sum(fpr)/len(fpr),sum(fnr)/len(fnr))
                    torch.save(model.state_dict(),"sentence_BERT_model_"+threshold+".pt")
                print("Current max F1 score= ",max_f1)
                clsf_report = pd.DataFrame(classification_report(y_true =tensor_val_y.t().cpu(), y_pred = predictions.t().cpu(), output_dict=True)).transpose()
                clsf_report.to_csv('newbert_'+str(e)+'report_result2.csv', index= True) 
                clsf_report = pd.DataFrame(classification_report(y_true =tensor_val_y.cpu(), y_pred = predictions.cpu(), output_dict=True,target_names=Technique_name)).transpose()
                clsf_report.to_csv('newbert_'+str(e)+'result2.csv', index= True)
    with open("loss_file_sen"+str(tensor_train_y.shape[0]),"wb") as fff:
                pickle.dump(losses,fff)
    fff.close()


if __name__ == "__main__":
    Technique_name_f=open("Data/Technique_name_80.txt","rb")
    Technique_name=pickle.load(Technique_name_f)
    Technique_name_f.close()
    des_emb_list=encode_description_finetuned(des_df,Technique_name)
    des_emb_list=torch.FloatTensor(des_emb_list).cuda()