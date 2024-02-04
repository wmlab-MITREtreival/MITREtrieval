from pandas import DataFrame
import torch
import pickle
import numpy as np
import pickle
import string
from nltk import sent_tokenize
from tqdm import tqdm
import codecs

import sys
sys.path.append("..") 
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
from BERT_description import MAX_LEN, CustomDataset,RobertaClass
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
model.eval()
model.cuda()
'''
 *sentences_segmentation(): Segment every documents into sentence list
 *@corpora: A list with mant CTI reports(document)
 *@tokenizer: Tokenizer to be used
 *@min_token: The lower bound words in a sentence
 *@return: sentence list
'''
def sentences_segmentation(corpora,tokenizer,min_token=0):
    
    segmented_documents = []
    for document in tqdm(corpora):
        segmented_document = []
        seg_document = sent_tokenize(document)

        ## remove sentences that are too short, the tokenized sentences should larger than min_token, otherwise are dropped
        for sentence in seg_document:
            tokenized_sentence = tokenizer.tokenize(sentence)
            if len(tokenized_sentence)>min_token:
                temp_sentence = tokenizer.convert_tokens_to_string(tokenized_sentence)
                ## if a whole sentence consists of punctations, it will be dropped
                if not all([j.isdigit() or j in string.punctuation for j in temp_sentence]):
                    segmented_document.append(temp_sentence)
            
        segmented_documents.append(segmented_document)
    
    return segmented_documents
'''
 *encode_description(): Encode each MITRE Description into embedding(Without Fine-tuned)
 *@description_df: Dataframe consists of Technique ID and their description
 *@Technique_list: Target Technique list
 *@return: Embedding of MITRE Description
'''
def encode_description(description_df:DataFrame, Technique_list:list):
    doc_sen_embedding=[]
    max_length=0
    for tech_iter in tqdm(Technique_list):
        for index,des_iter in description_df.iterrows():
            if(des_iter['ID']==tech_iter):
                input_ids=tokenizer(des_iter['description'])['input_ids']
                if len(input_ids)>512:
                    input_ids = input_ids[:512]
                tokens_tensor = torch.tensor([input_ids]).cuda()
                encoded_layers = model(tokens_tensor)
                embeddings_array = encoded_layers[0][0].cpu().detach().numpy()

                del encoded_layers
                del tokens_tensor
                
                if (np.shape(embeddings_array)[0]>max_length):
                    max_length=np.shape(embeddings_array)[0]
                doc_sen_embedding.append(embeddings_array)
    des_sen_embedding = np.zeros((len(doc_sen_embedding),max_length,768))
    for index,sen in enumerate(doc_sen_embedding):
        avg_sen = np.mean(sen,axis=0)
        doc_sen_embedding[index]=avg_sen
    return doc_sen_embedding

'''
 *encode_description(): Encode each MITRE Description into embedding(With Fine-tuned)
 *@description_df: Dataframe consists of Technique ID and their description
 *@Technique_list: Target Technique list
 *@return: Embedding of MITRE Description
'''
def encode_description_finetuned(description_df:DataFrame, Technique_list:list):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
    print(description_df)
    possible_labels = description_df.ID.unique()
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    description_df['label'] = description_df.ID.replace(label_dict)

    y = LabelBinarizer().fit_transform(description_df.label).tolist()
    df=[]
    sent_embedding=[]
    for tech_iter in tqdm(Technique_list):
        a=description_df.loc[description_df['ID']==tech_iter]
        df.append(a)
    df=pd.concat(df,ignore_index=True)
    MAX_LEN=512
    test_set = CustomDataset(df, tokenizer, MAX_LEN)
    TEST_BATCH_SIZE=1
    test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 1
                }
    test_loader = DataLoader(test_set, **test_params)
    model = RobertaClass()
    model.to(device='cuda')
    model.load_state_dict(torch.load("bert_description.pt"))
    model.eval()
    with torch.no_grad():
        for _, data in tqdm(enumerate(test_loader, 0)):
            ids = data['ids'].to(device='cuda', dtype = torch.long)
            mask = data['mask'].to(device='cuda', dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device='cuda', dtype=torch.long)
            outputs = model(ids, mask, token_type_ids)
            a=outputs[0].cpu().detach().numpy()
            a=np.squeeze(a,axis=0)
            sent_embedding.append(a)
            del a, outputs,ids, mask, token_type_ids
    del model
    return sent_embedding

'''
 * encode_sentence_doc()-Encode a given sentence or document
 * @text: Text in CTI report or in a sentence
 * @sentence : sentence or not
'''
def encode_sentence_doc(text,sentence=False):
    if sentence==True:
        input_ids = tokenizer(text)['input_ids']
            # if number of tokens in a sentence is large than 510
        if len(input_ids)>510:
            input_ids = input_ids[:512]

        tokens_tensor = torch.tensor([input_ids]).cuda()
        encoded_layers = model(tokens_tensor)

        embeddings_array = encoded_layers[0][0].cpu().detach().numpy()

        del encoded_layers
        del tokens_tensor
        embed = np.mean(embeddings_array,axis=0)
    else:
        corpora_=[]
        corpora_.append(text)
        segmented_documents = sentences_segmentation(corpora_,tokenizer)
        doc=segmented_documents[0]
        embed = []
        for sen in tqdm(doc):
            input_ids = tokenizer(sen)['input_ids']

            # if number of tokens in a sentence is large than 510
            if len(input_ids)>510:
                input_ids = input_ids[:512]

            tokens_tensor = torch.tensor([input_ids]).cuda()
            encoded_layers = model(tokens_tensor)
            embeddings_array = encoded_layers[0][0].cpu().detach().numpy()
            del encoded_layers
            del tokens_tensor
            sent_embed = np.mean(embeddings_array,axis=0)
            embed.append(sent_embed)
    return embed
'''
 * encode_doc()-Encode many document
 * @data_df: Dataframe consist of CTI reports
 * @return: Embedding of every CTI reports
'''
def encode_doc(data_df):
    length = []
    all_corpus=data_df['Text']
    segmented_documents = sentences_segmentation(all_corpus,tokenizer)
    for i in segmented_documents:
        length.append(len(i))
    print('max document length: ', np.max(length))
    print('mean document length: ', np.mean(length))
    print('standard deviation: ', np.std(length))
    doc_sen_embeddings = []
    with torch.no_grad():
        for doc in tqdm(segmented_documents):
            doc_sen_embedding = []
            for sen in tqdm(doc):
                input_ids = tokenizer(sen)['input_ids']
                
                # if number of tokens in a sentence is large than 510
                if len(input_ids)>510:
                    input_ids = input_ids[:512]

                tokens_tensor = torch.tensor([input_ids]).cuda()
                encoded_layers = model(tokens_tensor)

                embeddings_array = encoded_layers[0][0].cpu().detach().numpy()

                del encoded_layers
                del tokens_tensor

                doc_sen_embedding.append(embeddings_array)
            doc_sen_embeddings.append(doc_sen_embedding)
        
        doc_sen_avg_embeddings=[]
        for doc in doc_sen_embeddings:

            ## temp_doc shape [num of sentences in a doc, 768]
            temp_doc = []
            for sen in doc:
                avg_sen = np.mean(sen,axis=0)
                temp_doc.append(avg_sen)
            doc_sen_avg_embeddings.append(np.array(temp_doc))

   
    return doc_sen_avg_embeddings

def encode_sen_in_doc(data_df):

    all_corpus=data_df['Text']
    segmented_documents = sentences_segmentation(all_corpus,tokenizer)
    doc_sen_embeddings = []
    length=[]
    for i in segmented_documents:
        length.append(len(i))
    print('max document length: ', np.max(length))
    print('mean document length: ', np.mean(length))
    print('standard deviation: ', np.std(length))
    with torch.no_grad():
        for seg_iter in segmented_documents:
            doc_sen_embedding=[]
            #print(seg_iter)
            for index,sen in tqdm(enumerate(seg_iter)):
                sen=sen.replace("e ute","execute")
                sen=sen.replace("e utable","executable")
                sen=sen.replace("e uting","executing")
                sen=sen.replace("e ution","execution")
                input_ids = tokenizer(sen.lower())['input_ids']
                print(sen)
                print(input_ids)
                # if number of tokens in a sentence is large than 510
                if len(input_ids)>510:
                    input_ids = input_ids[:512]

                tokens_tensor = torch.tensor([input_ids]).cuda()
                encoded_layers = model(tokens_tensor)
                
                embeddings_array = encoded_layers[0][0].cpu().detach().numpy()
                del encoded_layers
                del tokens_tensor
                doc_sen_embedding.append(embeddings_array)
                print(embeddings_array)
            doc_sen_embeddings.append(doc_sen_embedding)
    return doc_sen_embeddings



if __name__=="__main__":
    with open('../dl_train_feature_50_clean',"rb") as f:
        train=pickle.load(f)
    encode_sen_in_doc(train)
