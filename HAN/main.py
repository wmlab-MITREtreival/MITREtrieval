import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import get_max_lengths, get_evaluation
from dataset import CustomDataset
from HAN import HierAttNet
from sklearn.metrics import f1_score, precision_score, recall_score,classification_report,multilabel_confusion_matrix
#from tensorboardX import SummaryWriter
import argparse
import shutil
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets.squeeze())
if __name__ == '__main__':
    training_params = {"batch_size": 16,
                       "shuffle": True,
                       "drop_last": True}
    test_params = {"batch_size": 16,
                   "shuffle": False,
                   "drop_last": False}
    
    with open("../dl_train_feature_50","rb") as f:
        df=pickle.load(f)
    with open("../dl_test_feature_50","rb") as ff:
        df1=pickle.load(ff)
    
    '''
    with open('../sentence_train',"rb") as f:
            df=pickle.load(f)
    with open('../sentence_valid',"rb") as of:
            df1=pickle.load(of)
    f.close()
    of.close()
    
    possible_labels = df.label.unique()
    possible_labels_val=df1.label.unique()

    #print(possible_labels)
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    df['list'] = df.label.replace(label_dict)
    label_dict_val = {}
    for index, possible_label in enumerate(possible_labels_val):
        label_dict_val[possible_label] = index
    df1['list'] = df1.label.replace(label_dict_val)
    with open("../Data/Technique_name_sentence.txt","rb") as fff:
        technique_name=pickle.load(fff)
    '''
    max_word_length, max_sent_length = get_max_lengths("../sentence_train")
    training_set = CustomDataset(dict_path="glove.6B.50d.txt",dataframe=df,max_length_sentences=max_sent_length, max_length_word=max_word_length)
    training_generator = DataLoader(training_set, **training_params)
    test_set = CustomDataset(dict_path="glove.6B.50d.txt",dataframe=df1,max_length_sentences=max_sent_length, max_length_word=max_word_length)
    test_generator = DataLoader(test_set, **test_params)
    
    model = HierAttNet(word_hidden_size=50, sent_hidden_size=50, batch_size=16, num_classes=training_set.num_classes,
                       pretrained_word2vec_path="glove.6B.50d.txt", max_sent_length= max_sent_length,max_word_length= max_word_length)
    if torch.cuda.is_available():
        model.cuda()
    #if torch.cuda.device_count() > 1:
        #model=torch.nn.DataParallel(model)
    #criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    best_loss = 1e-5
    best_epoch = 0
    model.train()
    
    #if torch.cuda.device_count() > 1:
        #model=torch.nn.DataParallel(model)

    model.cuda()
    num_iter_per_epoch = len(training_generator)
    for epoch in range(100):
        for iter, (feature, label) in tqdm(enumerate(training_generator)):
            if torch.cuda.is_available():
                feature = feature.cuda()
                #print(label.size)    
                label = label.cuda()
            optimizer.zero_grad()
            model._init_hidden_state()
            predictions = model(feature)
            #print(predictions)
            
            loss = loss_fn(predictions, label)
            loss.backward()
            optimizer.step()
            #training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
            '''
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))
            '''
            print("Epoch= ",epoch," Loss= ",loss)
            #writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
            #writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
        if epoch % 1 == 0:
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for te_feature, te_label in test_generator:
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    te_feature = te_feature.cuda()
                    te_label = te_label.cuda()
                with torch.no_grad():
                    model._init_hidden_state(num_sample)
                    te_predictions = model(te_feature)
                te_loss = loss_fn(te_predictions, te_label)
                loss_ls.append(te_loss * num_sample)
                te_label_ls.extend(te_label.cpu().detach().numpy().tolist())
                te_pred_ls.extend(te_predictions.cpu().detach().numpy().tolist())
            te_loss = sum(loss_ls) / test_set.__len__()
            '''
            for r in range(0, np.array(te_pred_ls, dtype="object").shape[0]):
                for c in range(0, np.array(te_pred_ls, dtype="object").shape[1]):
                    if te_pred_ls[r][c] > 0:
                        te_pred_ls[r][c] = 1
                    else:
                        te_pred_ls[r][c] = 0
            '''
            predict=[]
            for r in range(0, np.array(te_pred_ls, dtype="object").shape[0]):
                predict.append(torch.argmax(torch.FloatTensor(te_pred_ls[r])))
            #confusion=pd.DataFrame(multilabel_confusion_matrix(y_true = te_label_ls, y_pred = te_pred_ls).reshape(np.shape( te_pred_ls)[1],2*2))
            #confusion.to_csv('HAN_conf.csv',index=True)
            clsf_report = pd.DataFrame(classification_report(y_true = te_label_ls, y_pred = predict, output_dict=True,target_names=technique_name)).transpose()
            clsf_report.to_csv('HAN_result2_'+str(epoch)+'.csv', index= True)
            clsf_report = pd.DataFrame(classification_report(y_true = torch.FloatTensor(te_label_ls).t(), y_pred = predict, output_dict=True)).transpose()
            clsf_report.to_csv('HAN_result2_'+str(epoch)+'_report_based.csv', index= True)
            cm=multilabel_confusion_matrix(y_true = te_label_ls, y_pred = predict).reshape(34,2*2)
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
            model.train()
            #if te_loss + opt.es_min_delta < best_loss:
                #best_loss = te_loss
                #best_epoch = epoch
                #torch.save(model, opt.saved_path + os.sep + "whole_model_han")

            # Early stopping
            if epoch - best_epoch > 5 > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                

