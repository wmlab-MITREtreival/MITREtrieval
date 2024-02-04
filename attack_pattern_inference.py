from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from tqdm import tqdm
import torch
import re
import csv
import tensorflow as tf
from transformers import logging
import warnings
import multiprocessing as mp
from multiprocessing import Manager, Pool
from sentence_transformers import SentenceTransformer
import os
import math
from scipy import spatial
import itertools
from transformers import BertTokenizer
import pandas as pd
#from transformers import BertTokenizer, BertModel
#from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import transformers
import nltk
import json
from change_verb import change_verb
from ttp_dictionary import rule_based_ttp
from query_list import *

##############################################################################
# parameter setting
CH_VERB=True
USE_TACTIC=False
USE_RULE=True
v_threshold = 0.96 # best:0.95
o_threshold = 0.76 # best:0.76
##############################################################################

nltk.download("punkt")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings('ignore')
# print("gpu avaliable = ", tf.test.is_gpu_available())
# print(torch.version.cuda) # 10.2
# tensorflow_gpu-2.4.0

#print(transformers.__version__) # 2.5.1
model = SentenceTransformer('sentence-transformers/paraphrase-distilroberta-base-v2')
model.to(device='cuda')

"""
----------------------------------------------------------- 
FUNCTION get_authentication:
    get database's account and password from "data_source/neo4j_info.json"
PARAMETER USAGE :
    @ output : (dict) neo4j_info
               key = [url, account, password]
----------------------------------------------------------- 
"""
def get_authentication():
    with open('Data/neo4j_info.json') as f:
        neo4j_info = json.load(f)
    f.close()
    return neo4j_info

def word_embedding(sen):
    return model.encode(sen)

def get_embedding(): 
    all_vo_pair_df = pd.DataFrame() # = [ [att_patt_id, [ [verb, obj, v_verb, v_obj], [verb, obj, v_verb, v_obj], ... ],...  ]
    tactic_all_vo_pair_dict = dict()

    ###########################################################################
    # load word embedding and tactic verb-info
    if CH_VERB==False:
        print("load vo embedding without CH_Verb (change verb)")
        try: 
            with open('tactic_verb_obj_vector.pickle', 'rb') as f:
                all_vo_pair_df = pickle.load(f)
            f.close()
        except:
            # get ontology list with tactic info
            neo4j_info = get_authentication()
            greeter = Ontology(neo4j_info["url"],neo4j_info["account"], neo4j_info["password"])
            verb_object_list_tmp = greeter.query_all_Verb_Object_with_Tactic()
            greeter.close()
            global_vo=[]
            for i,row in verb_object_list_tmp.iterrows():
                local_vo=[]
                for verb, obj in row['vo_pair']:
                    v_verb = np.array(word_embedding(verb))
                    v_obj = np.array(word_embedding(obj))
                    local_vo.append([verb, obj, v_verb, v_obj])
                global_vo.append(local_vo)
            all_vo_pair_df['vo_pair'] = global_vo
            all_vo_pair_df['tactic'] = verb_object_list_tmp['tactic']
            all_vo_pair_df['srl_label'] = verb_object_list_tmp['srl_label']
            all_vo_pair_df['att_patt'] = verb_object_list_tmp['att_patt']
            # print(all_vo_pair_df.head())
            with open('tactic_verb_obj_vector.pickle', 'wb') as f:
                pickle.dump(all_vo_pair_df, f)

    elif CH_VERB==True:
        print("load vo embedding with CH_Verb")
        try: 
            with open('tactic_verb_obj_vector_CH_Verb.pickle', 'rb') as f:
                all_vo_pair_df = pickle.load(f)
            f.close()
        except:
            neo4j_info = get_authentication()
            greeter = Ontology(neo4j_info["url"],neo4j_info["account"], neo4j_info["password"])
            verb_object_list_tmp = greeter.query_all_Verb_Object_with_Tactic()
            greeter.close()
            global_vo=[]
            for i,row in verb_object_list_tmp.iterrows():
                local_vo=[]
                for verb, obj in row['vo_pair']:
                    verb = change_verb(verb)
                    v_verb = np.array(word_embedding(verb))
                    v_obj = np.array(word_embedding(obj))
                    local_vo.append([verb, obj, v_verb, v_obj])
                global_vo.append(local_vo)
            all_vo_pair_df['vo_pair'] = global_vo
            all_vo_pair_df['tactic'] = verb_object_list_tmp['tactic']
            all_vo_pair_df['srl_label'] = verb_object_list_tmp['srl_label']
            all_vo_pair_df['att_patt'] = verb_object_list_tmp['att_patt']
            
            with open('tactic_verb_obj_vector_CH_Verb.pickle', 'wb') as f:
                pickle.dump(all_vo_pair_df, f)

    ###########################################################################
    
    # make tactic_all_vo_pair_dict : {tactic_id:tactic_df}
    tactic_dict=dict()
    for i, row in all_vo_pair_df.iterrows():
        keys = row['tactic']
        for key in keys:
            values = tactic_dict.get(key)
            if (values is None):
                df = pd.DataFrame()
                df = df.append(row, ignore_index=True)
                tactic_dict[key] = df   
            else:
                tactic_dict[key] = tactic_dict[key].append(row, ignore_index=True)
            
    # for k,v in tactic_dict.items():
    #     print("{} {}".format(k, len(v)))
    tactic_all_vo_pair_dict = tactic_dict

    return tactic_all_vo_pair_dict

def remove_non_ascii(s):
    return "".join(c for c in s if ord(c)<128)

def change_threshold(v,o):
    global v_threshold
    global o_threshold
    v_threshold = v # 0.8
    o_threshold = o # 0.75

def cosine_similarity(v1,v2):
    return 1 - spatial.distance.cosine(v1,v2)

def vo_similarity(v_verb, v_obj, v_in_verb, v_in_obj):
    o_score = cosine_similarity(v_obj, v_in_obj)
    v_score = cosine_similarity(v_verb, v_in_verb)

    if o_score < o_threshold:
        return False
    if v_score < v_threshold:     
        return False
    return v_score*0.6 + o_score*0.4

def mapping(local_vo_pair_lst, v_in_verb, v_in_obj, match_vo_pair):
    for i,row in local_vo_pair_lst.iterrows():
        local_match=[]
        # 每個sentence可能有不同種srl分法->可能有多組的verb-obj, vo[0]=verb, vo[1]=obj, vo[2], v_verb, vo[3]=v_obj
        for vo in row['vo_pair']:
            simi_score=vo_similarity(vo[2], vo[3],  v_in_verb, v_in_obj)
            if simi_score != False:
                #append node = [score, [verb, obj] ]
                local_match.append([simi_score, [vo[0], vo[1] ]])
        if local_match !=[]:
            local_match = sorted(local_match, key = lambda x : x[0], reverse=True)
            print("match_vo=", local_match[0][1])
            print("###############")
            match_vo_pair.append(row['att_patt'])
            #ana_vo_pair.append([ local_match[0][1], row['att_patt'] ]) # local_match[0][1] : match verb-obj in ontology

def multi_vo_pair_mapping(tactic_all_vo_pair_dict, insert_vo_pair, tactic_list):
    # tactic_all_vo_pair_dict is same as vo_pair_embeddings
    selected_vo_pair = pd.DataFrame()
    # tactic narrow
    if USE_TACTIC==True:
        for t in tactic_list:
            selected_vo_pair = selected_vo_pair.append(tactic_all_vo_pair_dict[t.lower()], ignore_index=True)
    else:
        #selected_vo_pair = all_vo_pair_df
        for t in list(tactic_all_vo_pair_dict.keys()):
            selected_vo_pair = selected_vo_pair.append(tactic_all_vo_pair_dict[t.lower()], ignore_index=True)

    global all_vo_pair_lst
    total_match_vo_pair = []

    # for (all insert_vo pair)
    for vo in tqdm(insert_vo_pair):
        try:
            in_verb = vo[0]
            in_obj = vo[1]
        except:
            continue
        try:
            in_verb = remove_non_ascii(in_verb)
            in_obj = remove_non_ascii(in_obj)
            if CH_VERB==True:
                in_verb = change_verb(in_verb)
            v_in_verb = list(word_embedding(in_verb))
            v_in_obj = list(word_embedding(in_obj))
            # print("V: {}, O: {}".format(in_verb,in_obj))
            tmp_vo = []
            tmp_vo.append( [ vo[0], vo[1] ] )
        except:
            print("##### ontology word_embedding not work ######")
            continue

        manager = Manager()
        match_vo_pair = manager.list([]) # [ [ attpatt_id, score1, [verb,obj] ], [attpatt_id score2, [verb,obj] ], ... ]
        p_list = []
        cpu_cnt = mp.cpu_count()
        
        for i in range(cpu_cnt):
            start = int(len(selected_vo_pair)*i/cpu_cnt)
            end = int(len(selected_vo_pair)*(i+1)/cpu_cnt)

            p = mp.Process(target=mapping, args=(selected_vo_pair[start:end], v_in_verb,v_in_obj, match_vo_pair))
            p_list.append(p)
            p.start()

        for p in p_list:
            p.join()
        
        total_match_vo_pair+=list(match_vo_pair) # cover manager.list() to list
    return total_match_vo_pair

def attack_pattern_inference(vo_pair_embeddings, group=[], software=[], vo_pair=[], document="", srl_info=[], tactic_list=[]):
    
    # return three level technique, tactic
    # tech   = [level1:{tech1, ...}, level2:{tech1, ...}, level3:{tech1, ...}]
    # tactic = [level1:{T1, ...}, level2:{T1, ...}, level3:{T1, ...}]

    match_vo_pair = multi_vo_pair_mapping(vo_pair_embeddings, vo_pair, tactic_list)
    neo4j_info = get_authentication()
    greeter = Ontology(neo4j_info["url"],neo4j_info["account"], neo4j_info["password"])
    match_group = greeter.query_all_match_group(group)
    match_software = greeter.query_all_match_software(software)
    
    #print("############# inference #############")
    g_t, s_t, vo_t, g_vo_t, s_vo_t, g_s_vo_t = greeter.infer_attack_pattern(match_group, match_software, match_vo_pair)
    greeter.close()

    g_t = [ i[0].split('.')[0] for i in g_t]
    g_t = np.unique(g_t).tolist()

    s_t = [ i[0].split('.')[0] for i in s_t]
    s_t = np.unique(s_t).tolist()

    vo_t = [ i[0].split('.')[0] for i in vo_t]
    vo_t = np.unique(vo_t).tolist()

    s_vo_t = [ i[0].split('.')[0] for i in s_vo_t]
    s_vo_t = np.unique(s_vo_t).tolist()

    g_vo_t = [ i[0].split('.')[0] for i in g_vo_t]
    g_vo_t = np.unique(g_vo_t).tolist()

    g_s_vo_t = [ i[0].split('.')[0] for i in g_s_vo_t]
    g_s_vo_t = np.unique(g_s_vo_t).tolist()

    technique = { 'level1':g_t+s_t+vo_t, 'level2':g_vo_t+s_vo_t, 'level3':g_s_vo_t }

    if USE_RULE==True:
        total_ttp , must_ttp, rm_ttp = rule_based_ttp(document)
    else:
        must_ttp=[]
        rm_ttp = []

    print("vo_t = ", vo_t)
    print("g_t = ", g_t)
    print("s_t = ", s_t)
    print("g_vo_t = ", g_vo_t)
    print("s_vo_t = ", s_vo_t)
    print("g_s_vo_t = ", g_s_vo_t)
    print("must ttp = ", must_ttp)
    print("rm_ttp = ", rm_ttp)


    #return {'vo_t':vo_t, 'g_t':g_t, 's_t':s_t, 'g_vo_t':g_vo_t, 's_vo_t':s_vo_t, 'g_s_vo_t':g_s_vo_t, 'must_ttp':must_ttp}
    return g_t, s_t, vo_t, g_vo_t, s_vo_t, g_s_vo_t, must_ttp, rm_ttp

"""
# input with srl tag, not only vo_pair
def srl_mapping(local_vo_pair_lst, v_in_verb, v_in_obj, match_vo_pair):

    # result_df['tactic'] = tactic_lst
    # result_df['att_patt'] = id_lst
    # result_df['vo_pair'] = vo_lst
    # result_df['srl_label'] = srl_lst

    for i,row in local_vo_pair_lst.iterrows():
        local_match=[]
        # 每個sentence可能有不同種srl分法->可能有多組的verb-obj, vo[0]=verb, vo[1]=obj, vo[2], v_verb, vo[3]=v_obj
        for vo in row['vo_pair']:
            simi_score=vo_similarity(vo[2], vo[3],  v_in_verb, v_in_obj)
            if simi_score != False:
                #append node = [score, [verb, obj] ]
                local_match.append([simi_score, [vo[0], vo[1] ]])
        if local_match !=[]:
            local_match = sorted(local_match, key = lambda x : x[0], reverse=True)
            print("match_vo=", local_match[0][1])
            print("###############")
            match_vo_pair.append(row['att_patt'])
            #ana_vo_pair.append([ local_match[0][1], row['att_patt'] ]) # local_match[0][1] : match verb-obj in ontology

def srl_multi_vo_pair_mapping(srl_info, tactic_list):
    in_tactic = tactic_list
    #############################################################
    # extract vo in srl info
    insert_df = pd.DataFrame()
    sent_lst=[]
    sent_srl_lst=[]
    vo_lst=[]
    tactic_lst=[]
    for row in srl_info:
        # row[0]:sent, row[1]:srl info, row[2]:tactic
        sent = row[0]
        sent_srl=row[1]
        #tactic = row[2]
        local_vo=[]
        # for very srl parsing in one sentence
        for srl_parse in row[1]:
            v = None
            o=None
            for label in srl_parse:
                if 'V:' in label:
                    v = label.replace("V:", "").strip()
                if 'ARG1:' in label:
                    o = label.replace("ARG1:", "").strip()
                if v!=None and o!=None:
                    local_vo.append([v,o])
        
        local_vo = list(np.unique(local_vo, axis=0))
        vo_lst += local_vo
        sent_lst += [sent]*len(local_vo)
        sent_srl_lst += [sent_srl]*len(local_vo)
        #tactic_lst += [tactic]*len(local_vo)
        
    insert_df['vo_pair'] = vo_lst
    insert_df['sent'] = sent_lst
    insert_df['srl'] = sent_srl_lst
    #insert_df['tactic'] = tactic_lst
    #############################################################

    global all_vo_pair_df
    global tactic_all_vo_pair_dict

    selected_vo_pair = pd.DataFrame()
    # tactic narrow
    if USE_TACTIC==True:
        for t in in_tactic:
            selected_vo_pair = selected_vo_pair.append(tactic_all_vo_pair_dict[t.lower()], ignore_index=True)
    else:
        selected_vo_pair = all_vo_pair_df

    total_match_vo_pair = []
    # for (all insert_vo pair)
    for i, row in insert_df.iterrows():
        try:
            in_verb = row['vo_pair'][0]
            in_obj = row['vo_pair'][1]
            in_sent = row['sent']
            in_srl = row['srl']
            # in_tactic = row['tactic']
        except:
            print("nooooo")
            continue
        
        try:
            in_verb = remove_non_ascii(in_verb)
            in_obj = remove_non_ascii(in_obj)
            if CH_VERB==True:
                in_verb = change_verb(in_verb)
            v_in_verb = list(word_embedding(in_verb))
            v_in_obj = list(word_embedding(in_obj))
            print("V: {}, O: {}".format(in_verb,in_obj))
            #v_in_verb = np.array(word_embedding(in_verb)).reshape(1,-1)
            #v_in_obj = np.array(word_embedding(in_obj)).reshape(1,-1)
            tmp_vo = []
            tmp_vo.append( [ in_verb, in_obj ] )
        except:
            print("#####word_embedding not work######")
            continue

        manager = Manager()
        match_vo_pair = manager.list([]) # [ [ attpatt_id, score1, [verb,obj] ], [attpatt_id score2, [verb,obj] ], ... ]
        p_list = []
        cpu_cnt = mp.cpu_count()

        for i in range(cpu_cnt):
            start = int(len(selected_vo_pair)*i/cpu_cnt)
            end = int(len(selected_vo_pair)*(i+1)/cpu_cnt)
            p = mp.Process(target=srl_mapping, args=(selected_vo_pair.iloc[start:end, :], v_in_verb,v_in_obj, match_vo_pair))
            p_list.append(p)
            p.start()

        for p in p_list:
            p.join()

        total_match_vo_pair+=list(match_vo_pair) # cover manager.list() to list

    return total_match_vo_pair
"""

if __name__=='__main__':
    # attack_pattern_inference_1 : vo_pair+word_embedding
    # attack_pattern_inference_2 : description+example+tf-idf
    # attack_pattern_inference_3 : 1+2
    # result = get_tech_embedding(['t1566'])
    get_embedding()
    #doc = "As suggested, the first stage of the attack likely uses a spear phishing email to lure victims into opening an Excel file, which goes by the name “parliament_rew.xlsx”."
    #sentence_mapping(doc)
    result = multi_vo_pair_mapping([["exec", "various API functions such as NSCreateObjectFileImageFromMemory"]])
    print(result)

    neo4j_info = get_authentication()
    greeter = Ontology(neo4j_info["url"],neo4j_info["account"], neo4j_info["password"])
    for record in result:
        tech = greeter.transID_to_tech(record)
        vo_pair = greeter.transID_to_name(record)
        #print(tech)
        print(vo_pair)
    greeter.close()