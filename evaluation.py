import pickle
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from nltk import sent_tokenize
from nltk import word_tokenize
import codecs
import sys
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
from matplotlib.pyplot import MultipleLocator
from query_list import *
from sklearn.model_selection import train_test_split
def analzae_topic_classifier():
    with open('binary_bert','rb') as f:
        data=pickle.load(f)
    train_data, test_data = train_test_split(data, random_state=777, train_size=0.9)
    train_data, val_data = train_test_split(data, random_state=777, test_size=0.11)
    train_dataset = train_data.reset_index(drop=True)
    val_dataset=val_data.reset_index(drop=True)
    test_dataset=test_data.reset_index(drop=True)
    print(train_dataset,val_dataset,test_dataset)
    word=[]
    for ind, iter in test_dataset.iterrows():
        word.append(len(word_tokenize(iter['Text'])))
    print(sum(word)/len(word),np.std(word))
    '''
    mitre=[]
    mitre_word=[]
    news=[]
    news_word=[]
    for i,df_iter in data.iterrows():
        if df_iter['label']==0:
            news.append(df_iter['Text'])
        else:
            mitre.append(df_iter['Text'])
    for mitre_iter in mitre:
        mitre_word.append(len(word_tokenize(mitre_iter)))
    for news_iter in news:
        news_word.append(len(word_tokenize(news_iter)))
    print(sum(mitre_word)/len(mitre_word),np.std(mitre_word),sum(news_word)/len(news_word),np.std(news_word))
    '''
def train_val_test_sentence():
    with open("dl_train_feature_50","rb") as f:
        train=pickle.load(f)
    with open("dl_test_feature_50","rb") as f:
        test=pickle.load(f)
    f.close()
    val=train.loc[539:]
    train=train.loc[0:539]
    sen=[]
    word=[]
    for i,df_iter in val.iterrows():
        sen_list=sent_tokenize(df_iter['Text'])
        sen.append(len(sen_list))
        for sen_iter in sen_list:
            word.append(len(word_tokenize(sen_iter)))
    print(sum(sen)/len(sen),np.std(sen))
def dataset_sentence_evaluate():
    #sentence/word of every report in dataset
    all_data=pd.read_csv('Data/training_data_original_refine1.csv')
    #all_data1=pd.read_csv('Data/training_data_original.csv',encoding = "ISO-8859-1")
    #print(all_data.iloc[:613])
    train_data_df=all_data.iloc[0:]
    train_data_df.drop(train_data_df.columns[[1, 2, 3,4,5,6,7,8,9,10,11,12,13,14]], axis = 1, inplace = True)
    train_data_df= train_data_df.loc[:, (train_data_df != 0).any(axis=0)]
    train_data_df= train_data_df.reset_index()
    label=train_data_df
    

    label=label.iloc[:,1:]
    
    label.fillna(0)
    label.replace(np.nan, 0, inplace=True)
    label.replace(np.inf, 0, inplace=True)
    label= label.loc[:, (label != 0).any(axis=0)]
    train_data_df1=all_data.iloc[0:613]
    print(train_data_df1,type(train_data_df))
    total_sent=[]
    total_word=[]
    for i, report in train_data_df1.iterrows():
        sentence_in_doc=sent_tokenize(str(report['Text']))
        total_sent.append(len(sentence_in_doc))
        for sent in sentence_in_doc:
            words=word_tokenize(sent)
            total_word.append(len(words))
    print(total_sent,len(total_sent))
    print( sum(total_sent) / len(total_sent),np.std(total_sent))
    print(sum(total_word) / len(total_word),np.std(total_word))

def distinct_technique():
    #Technique Distribution Real
    all_data=pd.read_csv('Data/training_data_original_refine1.csv')
    #all_data1=pd.read_csv('Data/training_data_original.csv',encoding = "ISO-8859-1")
    #print(all_data.iloc[:613])
    train_data_df=all_data.iloc[0:]
    train_data_df.drop(train_data_df.columns[[1, 2, 3,4,5,6,7,8,9,10,11,12,13,14]], axis = 1, inplace = True)
    train_data_df= train_data_df.loc[:, (train_data_df != 0).any(axis=0)]
    train_data_df= train_data_df.reset_index()
    label=train_data_df
    

    label=label.iloc[:,1:]
    
    label.fillna(0)
    label.replace(np.nan, 0, inplace=True)
    label.replace(np.inf, 0, inplace=True)
    label= label.loc[:, (label != 0).any(axis=0)]
    #label.reset_index()
    print(label,np.shape(label))

    technique_name=list(label.columns)[1:]
    print(technique_name)
    TTP_ana=[0]*(np.shape(label)[1]-1)
    train_data_df1=train_data_df.iloc[:,0].astype(str)
    temp=[]
    for i,each_TTP_label in enumerate(train_data_df1):
        #print(i)
        temp=label.iloc[i,1:]
        for j,content in enumerate(temp):
            #print(j)
            if(content==1):
                TTP_ana[j]=TTP_ana[j]+1
    print(TTP_ana)
    
    report_num=[0]*np.shape(label)[0]
    for i,each_report in label.iterrows():
        for inin,itit in enumerate(each_report):
            if(itit==1):
                report_num[i]=report_num[i]+1
    print(report_num,np.shape(report_num))
    print( sum(report_num) / len(report_num),np.std(report_num))
    print( sum(TTP_ana) / len(TTP_ana),np.std(TTP_ana))
    list1, list2=zip(*sorted(zip(TTP_ana,technique_name),reverse=True))
    plt.figure(figsize=(25,10),dpi=100,linewidth = 2)
    plt.plot(list2,list1,'X-')

    plt.xlabel('Technique')
    plt.ylabel('Report num')
    plt.xticks(rotation=90)
    plt.title("Technique Distribution")
    plt.savefig("Technique Distribution.jpg")


def Technique_Distribution(threshold:str):
    
    with open("dl_train_feature_"+threshold+"_clean","rb") as f:
        train_data=pickle.load(f)
    f.close()
    with open("dl_test_feature_"+threshold+"_clean","rb") as f:
        test_data=pickle.load(f)
    f.close()
    Technique_name_f=open("Data/Technique_name_"+threshold+".txt","rb")
    Technique_name=pickle.load(Technique_name_f)
    Technique_name_f.close()

    total_data=pd.concat([train_data,test_data],ignore_index=True)
    print(total_data,np.shape(total_data))
    tech_list=[0]*len(Technique_name)
    for index,iter_train in test_data.iterrows():
        for indexes,iter_list in enumerate(iter_train['list']):
            if iter_list==1:
                tech_list[indexes]=tech_list[indexes]+1
    n, bins, patches=plt.hist(tech_list)
    

    plt.xlabel("Techniques Appear times")
    plt.ylabel("Number of Technique")
    plt.title("Technique Distribution")
    plt.savefig("Technique Distribution.jpg")
    print(tech_list)
import csv
def createHTML(texts, fileName):
    """
    Creates a html file with text heat.
	weights: attention weights for visualizing
	texts: text on which attention weights are to be visualized
    """
    #fileName = "visualization/"+fileName

    texts=[texts]
    fOut = open(fileName, "w", encoding="utf-8")
    part1 = """
    <html lang="en">
    <head>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <style>
    body {
    font-family: Sans-Serif;
    }
    </style>
    </head>
    <body>
    <h3>
    Heatmaps
    </h3>
    </body>
    <script>
    """
    part2 = """
    var color = "255,0,0"
    var tokens = any_text[0].split(" ");
    var heat_text = "<b>CTI report Example:</b><br>";

    var space = "";
    for (var i = 0; i < tokens.length; i++) {
    heat_text += space + tokens[i] ;
    if (space == "") {
    space = " ";
    }
    }
    //heat_text += "<p>";
    document.body.innerHTML += heat_text;
    
    </script>
    </html>"""
    putQuote = lambda x: "\"%s\""%x
    textsString = "var any_text = %s;\n"%(texts)
    fOut.write(part1)
    fOut.write(textsString)
    fOut.write(part2)
    fOut.close()
  
    return
def intermediate_result2(threshold:str):
    Technique_name_f=open("Data/Technique_name_"+threshold+".txt","rb")
    Technique_name=pickle.load(Technique_name_f)
    Technique_name_f.close()
    with open('MITretrieval_'+threshold+'.csv',newline='') as csvfile:
        rows=csv.reader(csvfile)
        fn_list=[]
        fp_list=[]
        tp_list=[]
        for index,row in enumerate(rows):
            if index==0:
                continue
            tp=int(row[4])
            fp=int(row[2])
            fn=int(row[3])
            fn_list.append(fn)
            fp_list.append(fp)
            tp_list.append(tp)
    with open('Fusion_conf6.csv',newline='') as csvfile:
        rows=csv.reader(csvfile)
        fn_list_fu=[]
        fp_list_fu=[]
        tp_list_fu=[]
        for index,row in enumerate(rows):
            if index==0:
                continue
            tp=int(row[4])
            fp=int(row[2])
            fn=int(row[3])
            fn_list_fu.append(fn)
            fp_list_fu.append(fp)
            tp_list_fu.append(tp)
    with open('rcatt_conf.csv',newline='') as csvfile:
        rows=csv.reader(csvfile)
        fn_list_rc=[]
        tp_list_rc=[]
        for index,row in enumerate(rows):
            if index==0:
                continue
            tp=int(row[4])
            fp=row[2]
            fn=int(row[3])
            fn_list_rc.append(fn)
            tp_list_rc.append(tp)
    plt.figure(figsize=(25,10),dpi=100,linewidth = 2)
    new_fn_list,new_fn_list_rc=zip(*sorted(zip(fp_list,fp_list_fu )))
    new_fn_list,new_Technique_name=zip(*sorted(zip(fp_list, Technique_name)))
    print(fn_list,new_fn_list )
    plt.plot(new_Technique_name,new_fn_list,'o-')
    #plt.plot(Technique_name,fn_list_fu,'s-')
    plt.plot(new_Technique_name,new_fn_list_rc,'X-')
    plt.legend(['Before','After'])
    plt.xlabel("Technique")
    plt.xticks(rotation=90)
    plt.ylabel("FP")
    plt.title("False Positive")
    plt.savefig("FP.jpg")
def intermediate_result(threshold:str):
    Technique_name_f=open("Data/Technique_name_"+threshold+".txt","rb")
    Technique_name=pickle.load(Technique_name_f)
    Technique_name_f.close()
    with open('MITretrieval_'+threshold+'.csv',newline='') as csvfile:
        rows=csv.reader(csvfile)
        fn_list=[]
        fp_list=[]
        tp_list=[]
        for index,row in enumerate(rows):
            if index==0:
                continue
            tp=int(row[4])
            fp=int(row[2])
            fn=int(row[3])
            fn_list.append(fn)
            fp_list.append(fp)
            tp_list.append(tp)
    with open('Fusion_conf6.csv',newline='') as csvfile:
        rows=csv.reader(csvfile)
        fn_list_fu=[]
        fp_list_fu=[]
        tp_list_fu=[]
        for index,row in enumerate(rows):
            if index==0:
                continue
            tp=int(row[4])
            fp=int(row[2])
            fn=int(row[3])
            fn_list_fu.append(fn)
            fp_list_fu.append(fp)
            tp_list_fu.append(tp)
    with open('rcatt_conf.csv',newline='') as csvfile:
        rows=csv.reader(csvfile)
        fn_list_rc=[]
        tp_list_rc=[]
        for index,row in enumerate(rows):
            if index==0:
                continue
            tp=int(row[4])
            fp=row[2]
            fn=int(row[3])
            fn_list_rc.append(fn)
            tp_list_rc.append(tp)
    with open('HAN/HAN_conf.csv',newline='') as csvfile:
        rows=csv.reader(csvfile)
        fn_list_han=[]
        tp_list_han=[]
        for index,row in enumerate(rows):
            if index==0:
                continue
            tp=int(row[4])
            fp=row[2]
            fn=int(row[3])
            fn_list_han.append(fn)
            tp_list_han.append(tp)
    plt.figure(figsize=(25,10),dpi=100,linewidth = 2)
    new_fn_list,new_fn_list_rc=zip(*sorted(zip(fn_list, fn_list_rc)))
    new_fn_list,new_fn_list_han=zip(*sorted(zip(fn_list, fn_list_han)))
    new_fn_list,new_Technique_name=zip(*sorted(zip(fn_list, Technique_name)))
    print(fn_list,new_fn_list,new_fn_list_han )
    plt.plot(new_Technique_name,new_fn_list,'o-')
    #plt.plot(Technique_name,fn_list_fu,'s-')
    plt.plot(new_Technique_name,new_fn_list_rc,'X-')
    plt.plot(new_Technique_name,new_fn_list_han,'s-')
    plt.legend(['MITretrieval','rcATT','HAN'])
    plt.xlabel("Technique")
    plt.xticks(rotation=90)
    plt.ylabel("FN")
    plt.title("False Negative")
    plt.savefig("FN.jpg")
def case_study(threshold:str):
    with open("Predict_ans_"+threshold,"rb") as fff:
        mitretrieval=pickle.load(fff)
    #print(mitretrieval[99])
    Technique_name_f=open("Data/Technique_name_"+threshold+".txt","rb")
    Technique_name=pickle.load(Technique_name_f)
    Technique_name_f.close()
    with open("dl_test_feature_"+threshold+"_clean","rb") as of:
        test_=pickle.load(of)
    for index,ititit in enumerate(mitretrieval[51]):
        if ititit==1:
            print(Technique_name[index])
    print("-----------------------------")
    print(test_.iloc[51,:]['Text'])
    createHTML(test_.iloc[51,:]['Text'],"case_study.html")
    for indexeses, it in enumerate(test_.iloc[51,:]['list']):
        if(it==1):
            print(Technique_name[indexeses])
import csv
#Measuring specific range performance(6-40、4-80)
def for_cycarrier():
    with open('Fusion_conf_80.csv',newline='') as csvfile:
        rows=csv.reader(csvfile)
        fn_list_fu=[]
        fp_list_fu=[]
        tp_list_fu=[]
        tn_list_fu=[]
        for index,row in enumerate(rows):
            if index==0:
                continue
            tp=int(row[4])
            fp=int(row[2])
            fn=int(row[3])
            tn=int(row[1])
            fn_list_fu.append(fn)
            fp_list_fu.append(fp)
            tn_list_fu.append(tn)
            tp_list_fu.append(tp)
        print(sum(fp_list_fu)/(sum(fp_list_fu)+sum(tn_list_fu)),sum(fp_list_fu),sum(tn_list_fu))
def specific_range_performance():
     
    all_data=pd.read_csv('Data/training_data_original_refine1.csv')
    #all_data1=pd.read_csv('Data/training_data_original.csv',encoding = "ISO-8859-1")
    #print(all_data.iloc[:613])
    train_data_df=all_data.iloc[0:]
    train_data_df.drop(train_data_df.columns[[1, 2, 3,4,5,6,7,8,9,10,11,12,13,14]], axis = 1, inplace = True)
    train_data_df= train_data_df.loc[:, (train_data_df != 0).any(axis=0)]
    train_data_df= train_data_df.reset_index()
    label=train_data_df
    

    label=label.iloc[:,1:]
    
    label.fillna(0)
    label.replace(np.nan, 0, inplace=True)
    label.replace(np.inf, 0, inplace=True)
    label= label.loc[:, (label != 0).any(axis=0)]
    #label.reset_index()
    print(label,np.shape(label))

    technique_name=list(label.columns)[1:]
    print(technique_name)
    TTP_ana=[0]*(np.shape(label)[1]-1)
    train_data_df1=train_data_df.iloc[:,0].astype(str)
    temp=[]
    for i,each_TTP_label in enumerate(train_data_df1):
        #print(i)
        temp=label.iloc[i,1:]
        for j,content in enumerate(temp):
            #print(j)
            if(content==1):
                TTP_ana[j]=TTP_ana[j]+1
    print(TTP_ana)
    file_=open('newbert_20result2.csv')
    file_1=open('MITretrieval_6.csv')
    tp=[]
    fp=[]
    fn=[]
    csvreader=csv.reader(file_)
    csvreader1=list(csv.reader(file_1))
    print(csvreader1)
    for index,row in enumerate(csvreader):
        for i,tech_iter in enumerate(technique_name):
            if(row[0]==tech_iter and TTP_ana[i]>80 ):
                tp.append(int(csvreader1[index][4]))
                fn.append(int(csvreader1[index][3]))
                fp.append(int(csvreader1[index][2]))
    print(tp)
    precision=sum(tp)/(sum(tp)+sum(fp))
    recall=sum(tp)/(sum(tp)+sum(fn))
    f2=5*precision*recall/((4*precision)+recall)
    print(precision,recall,f2)
if __name__=="__main__":
    train_val_test_sentence()