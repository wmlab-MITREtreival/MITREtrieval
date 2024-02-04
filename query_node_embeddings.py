import pickle
from sys import meta_path
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy import spatial
import numpy as np
import torch
import heapq

# https://www.sbert.net/
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
metapath_df = pd.DataFrame()

def load_metapath_df():
    global metapath_df
    with open("data_source/metapath_df.pickle", 'rb') as f:
        metapath_df = pickle.load(f)
        """
            result_df["uid"] = uid_list
            result_df["metapath_emb"] = metapath_list
            result_df["sentbert_emb"] = sentbert_list
            result_df["object"] = obj_list
        """
    f.close()

def cosine_similarity(v1,v2):
    return 1 - spatial.distance.cosine(v1,v2)

def query_node_embeddings(q_obj, threshold):
    global metapath_df
    global model

    q_emb = model.encode([q_obj])[0]
    node_embs = []
    for i, row in metapath_df.iterrows():
        score = cosine_similarity(q_emb, row["sentbert_emb"])
        if score >= threshold:
            print("{}, {}".format(score, row["object"]))
            node_embs.append(row["metapath_emb"])
    
    print("match node num : ", len(node_embs))
    # no match node
    if len(node_embs)==0:
        return torch.Tensor([0]*128)
    
    return torch.mean(torch.Tensor(node_embs), 0)

def query_top_n_node_embeddings(q_obj, threshold):
    global metapath_df
    global model
    top_n=5

    q_emb = model.encode([q_obj])[0]
    node_embs = []
    scores = []
    for i, row in metapath_df.iterrows():
        score = cosine_similarity(q_emb, row["sentbert_emb"])
        if score >= threshold:
            print("{}, {}".format(score, row["object"]))
            node_embs.append(row["metapath_emb"])
            scores.append(score)
    
    # no match node
    if len(node_embs)==0:
        return torch.Tensor([0]*128)
    elif len(node_embs)<=top_n:
        return torch.max(torch.Tensor(node_embs), 0)[0]
    else:
        #top_n_scores = heapq.nlargest(top_n, scores)
        top_n_idx = list(map(scores.index, heapq.nlargest(top_n, scores)))

        top_n_embs = []
        for idx in top_n_idx:
            top_n_embs.append(node_embs[idx])

        return torch.max(torch.Tensor(top_n_embs), 0)[0]

if __name__=='__main__':
    q_obj = "spear-phishing document sent to the japanese support team . spear-phishing document sent to the chinese support team support teams are used to receiving requests from customers, and emails that contain screen captures from users may also be commonplace for these teams."
    threshold = 0.5

    load_metapath_df()
    emb = query_node_embeddings(q_obj, threshold)
    print(emb.shape)
    print(type(emb))