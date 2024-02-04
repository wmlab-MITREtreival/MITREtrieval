from query_list import Ontology
from transformers import BertTokenizer, BertModel
import torch
import re
import csv
import tensorflow as tf
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

verb_object_list=[]
def get_embedding():
    global verb_object_list
    try:
        with open('verb_obj_vector.pickle', 'rb') as f:
            verb_object_list = pickle.load(f)
    except:
        # get ontology list
        greeter = Ontology("bolt://140.115.54.74:7687", "neo4j", "wmlab")
        verb_object_list_tmp = greeter.query_all_Verb_Object()
        greeter.close()
        for record in verb_object_list_tmp:
            tech_id = record[0]
            verb = record[1]
            obj = record[2]
            w_verb = record[3]
            w_obj = record[4]
            w_total = record[5]
            v_verb = np.array(word_embedding(verb)).reshape(1,-1)
            v_obj = np.array(word_embedding(obj)).reshape(1,-1)
            verb_object_list.append([tech_id, verb, obj, w_verb, w_obj, w_total, v_verb, v_obj])
        
        with open('verb_obj_vector.pickle', 'wb') as f:
            pickle.dump(verb_object_list, f)


def ontology_verb_obj_finding(in_verb, in_obj):
    result = [] # [ [ttp, score], [ttp, score], ...]

    # set word embedding model
    v_in_verb = np.array(word_embedding(in_verb)).reshape(1,-1)
    v_in_obj = np.array(word_embedding(in_obj)).reshape(1,-1)

    # verb_object_list = [tech_id, verb, obj, w_verb, w_obj, w_total]
    for record in verb_object_list:
        tech_id = record[0]
        verb = record[1]
        obj = record[2]
        w_verb = record[3]
        w_obj = record[4]
        w_total = record[5]
        v_verb = record[6]
        v_obj = record[7]

        verb_score = cosine_similarity(v_verb, v_in_verb)
        obj_score = cosine_similarity(v_obj, v_in_obj)
        if verb_score > 0.85 and obj_score > 0.8:
            #print("tech_id = ", tech_id)
            #print("verb = ", verb)
            #print("obj = ", obj)
            #print("-----------------------")
            score = verb_score*100*0.6 + obj_score*100*0.4
            #result.append([tech_id, score])
            result.append([tech_id, score])

    if result == None:
        return None

    result_sort = sorted(result, key = lambda x : x[1], reverse=True)
    Only_TTP=[]
    for TTP_each in result_sort:
        if '.' in TTP_each[0]:
            item=TTP_each[0].split('.')[0]
            Only_TTP+=[item]
        else:
            Only_TTP+=[TTP_each[0]]
            item = TTP_each[0]
        return [item]
    #Only_TTP=list(set(Only_TTP))
    if Only_TTP == []:
        return []
    return Only_TTP[0]
        
    

def word_embedding(sen):
    global tokenizer
    global model

    marked_text = "[CLS] " + sen + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    #print (tokenized_text)nvcc--version
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [1] * len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    sentence_embedding = sentence_embedding.numpy().tolist() 
    return sentence_embedding


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',
                                    output_hidden_states = True, # Whether the model returns all hidden-states.
                                    )
model.eval()
get_embedding()