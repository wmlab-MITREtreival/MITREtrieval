import json
import os 
import numpy as np 
import pandas as pd
import pickle
import codecs
import sys
import re
from nltk import sent_tokenize, word_tokenize
from query_list import *
import matplotlib.pyplot as plt 
import codecs
import sys
import json
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
#sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

def get_authentication():
    with open('Data/neo4j_info.json') as f:
        neo4j_info = json.load(f)
    f.close()
    return neo4j_info

def technique_to_tactic(ttp,ttp_v10):
    neo4j_info = get_authentication()
    greeter = Ontology(neo4j_info["url"],neo4j_info["account"], neo4j_info["password"])
    # greeter = Ontology("bolt://140.115.54.74:7687", "neo4j", "wmlab")
    print(greeter.get_technique2tactic_dict())
    tech2tac=greeter.get_technique2tactic_dict()
    tactic_dict={'ta0001':0,'ta0002':0,'ta0003':0,'ta0004':0,'ta0005':0,'ta0006':0,'ta0007':0,'ta0008':0,'ta0009':0,'ta0010':0,'ta0011':0,'ta0040':0}
    for index,ttp_iter in enumerate(ttp):
        for i, (tech,tactic) in enumerate(tech2tac.items()):
            if(ttp_iter!=None and ttp_iter.lower()==tech):
                for tactic_iter in tactic:
                    #print(tactic_iter,index)
                    tactic_dict[tactic_iter]=tactic_dict[tactic_iter]+ttp_v10[index]

    tactic_array=[0]*12
    for i, (tactic,tactic_num) in enumerate(tactic_dict.items()):
        tactic_array[i]=tactic_num
    print(tactic_array)
    print(sum(tactic_array) / len(tactic_array),np.std(tactic_array))
def sent_tactic(tech):
    #print(tech,len(tech))
    greeter = Ontology("bolt://140.115.54.74:7687", "neo4j", "wmlab")
    tech2tac=greeter.get_technique2tactic_dict()
    sent_tac=[]
    for tech_iter in tech:
        for i, (tech,tactic) in enumerate(tech2tac.items()):
            if(tech_iter!=None and tech_iter.lower()==tech):
                sent_tac.append(len(tactic))
    print(sent_tac)
    print(sum(sent_tac)/len(sent_tac),np.std(sent_tac))
def one_v62v10(technique):
    #print(tech,len(tech))
    with open("v6_to_v10_dict","rb") as f:
        dictv6=pickle.load(f)
    f.close()
    sent_tac=[]
    for i, (v6,v10) in enumerate(dictv6.items()):
        if(technique==v6):
            return v10
    return None
def build_dataset():
    cycarrier_report_path="mitre_tag"
    files=[f for f in os.listdir(cycarrier_report_path) if os.path.isfile(os.path.join(cycarrier_report_path,f))]

    column_name=["Text","Technique"]
    cycarrier_report=pd.DataFrame(columns=column_name)
    files.sort()
    
    num_of_TTP=[]
    num_of_report=0
    all_TTP=[]
    all_TTP_dict={}
    all_sentence=[]
    num_of_sent_accept=0
    tech=[]
    for index,file_iterator in enumerate(files):
        with open(cycarrier_report_path+"/"+ file_iterator) as f:
            json_content=json.load(f)
        #print(type(json_content))
        num_of_report=num_of_report+1
        #print(json_content['sentense_list'])
        report=""
        TTPs=""
        
        #print(json_content['id'])
        for data_in_json in json_content['sentense_list']:
            data_in_json['sentense']=data_in_json['sentense'].encode("ascii","ignore")
            data_in_json['sentense']=data_in_json['sentense'].decode()
            report=report+data_in_json['sentense']

        for data_in_json in json_content['detection_list']:
            if 'decision' in data_in_json and data_in_json['decision'] =='accept':
                num_of_sent_accept=num_of_sent_accept+1
                TTPs=TTPs+data_in_json['rule_id']
                v10ttp=one_v62v10(data_in_json['rule_id'])
                if(v10ttp==None or v10ttp=='None'):
                    continue
                elif v10ttp=="T1021/003":
                    v10ttp="T1021"
                elif v10ttp=="T1574/007":
                    v10ttp="T1574"
                all_sentence.append(data_in_json['sentense'])
                all_TTP.append(v10ttp)
    dataset=pd.DataFrame({'Text': all_sentence,'label': all_TTP})
    print(dataset)
    with open("cycarrier_data","wb") as f:
        pickle.dump(dataset,f)
def data_check():
    with open("cycarrier_data","rb") as f:
        dataset=pickle.load(f)
    for iter,data in dataset.iterrows():
        print(data['Text'])
def data_process():
    cycarrier_report_path="mitre_tag"
    files=[f for f in os.listdir(cycarrier_report_path) if os.path.isfile(os.path.join(cycarrier_report_path,f))]

    column_name=["Text","Technique"]
    cycarrier_report=pd.DataFrame(columns=column_name)
    files.sort()
    num_of_sentences=[]
    num_of_words=[]
    num_of_TTP=[]
    num_of_report=0
    all_TTP=[]
    all_TTP_dict={}
    all_sentence=[]
    num_of_sent_accept=0
    tech=[]
    for index,file_iterator in enumerate(files):
        with open(cycarrier_report_path+"/"+ file_iterator) as f:
            json_content=json.load(f)
        #print(type(json_content))
        num_of_report=num_of_report+1
        #print(json_content['sentense_list'])
        report=""
        TTPs=""
        
        #print(json_content['id'])
        for data_in_json in json_content['sentense_list']:
            data_in_json['sentense']=data_in_json['sentense'].encode("ascii","ignore")
            data_in_json['sentense']=data_in_json['sentense'].decode()
            report=report+data_in_json['sentense']
        sentence_list=sent_tokenize(report)
        num_of_sentences.append(len(sentence_list))
        word_list=report.split()
        num_of_words.append(len(word_list))
        for data_in_json in json_content['detection_list']:
            if 'decision' in data_in_json and data_in_json['decision'] =='accept':
                num_of_sent_accept=num_of_sent_accept+1
                TTPs=TTPs+data_in_json['rule_id']
                data_in_json['sentense']=re.sub('\b.','',data_in_json['sentense'])
                all_sentence.append(data_in_json['sentense'])
                all_TTP.append(data_in_json['rule_id'])
                tech.append(data_in_json['rule_id'])
                if data_in_json['rule_id'] in all_TTP_dict:
                    all_TTP_dict[data_in_json['rule_id']]=all_TTP_dict[data_in_json['rule_id']]+1
                else:
                    all_TTP_dict[data_in_json['rule_id']]=1
        all_TTP=list( dict.fromkeys(all_TTP))
        num_of_TTP.append(len(all_TTP))
        cycarrier_report.loc[index]=[report,TTPs]
    dict(sorted(all_TTP_dict.items(),key=lambda item :item[1]))

    all_words=[]
    for sent in all_sentence:
        all_words.append(len(word_tokenize(sent)))

    print(sum(all_words) / len(all_words),np.std(all_words))
    #print(num_of_sent_accept)
    #print(all_TTP_dict,len(all_TTP_dict),num_of_sent_accept)
    res=[0]*len(all_TTP_dict)
    for i,(dict_iter,num_ttp) in enumerate(all_TTP_dict.items()):
        res[i]=num_ttp
    
    #print(sum(res)/len(all_TTP_dict),np.std(res))
    sent_tactic(tech)
    ''' 
    pos=np.arange(len(all_TTP_dict.keys()))
    width=1.0
    ax = plt.axes() 
    ax.set_xticks(pos + (width/2)) 
    ax.set_xticklabels(all_TTP_dict.keys()) 
    plt.bar(all_TTP_dict.keys(), all_TTP_dict.values(), width, color='g') 
    plt.savefig("TTP_cycarrier.jpg")

    cycarrier_report.to_csv("New Report.csv")
    '''
    with open("v6_to_v10_dict","rb") as f:
        dictv6=pickle.load(f)
    f.close()
    ttp=[]

    for i,(dict_iter,num_ttp) in enumerate(all_TTP_dict.items()):
        #print(dict_iter)
        for j,(v6,v10) in enumerate(dictv6.items()):
            if (dict_iter==v6):
                if v10=="T1021/003":
                    dictv6[v6]="T1021"
                    v10="T1021"
                elif v10=="T1574/007":
                    dictv6[v6]="T1574"
                    v10="T1574"
                elif v10=='None':
                    continue
                elif v10==None:
                    continue
                ttp.append(v10)
     
    print(list(set(ttp)),len(list(set(ttp))))
    ttp_new=list(set(ttp))
    ttp_v10=[0]*len(list(set(ttp)))
    for i,(dict_iter,num_ttp) in enumerate(all_TTP_dict.items()):
        #print(dict_iter)
        for j,(v6,v10) in enumerate(dictv6.items()):
            if (dict_iter==v6 and v10!=None):
                
                index=ttp_new.index(v10)
                if(v10=='T1021'):
                    print("in=",index)
                ttp_v10[index]=ttp_v10[index]+num_ttp
    print(sum(ttp_v10) / len(ttp_v10),np.std(ttp_v10))
    print("39=",ttp_v10[39])
    dictionary = dict(zip(ttp_new, ttp_v10))
    print(dictionary)
    print(ttp_new,ttp_v10,len(ttp_new),len(ttp_v10))
    list1, list2=zip(*sorted(zip(ttp_v10,ttp_new),reverse=True))
    plt.figure(figsize=(25,10),dpi=100,linewidth = 2)
    plt.plot(list2,list1,'X-')

    plt.xlabel('Technique')
    plt.ylabel('Sentence num')
    plt.xticks(rotation=90)
    plt.title("Technique Distribution")
    plt.savefig("Technique Distribution of cycarrier.jpg")
    #technique_to_tactic(ttp_new,ttp_v10)
if __name__=="__main__":
    data_check()