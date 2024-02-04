import pandas as pd
from nltk import sent_tokenize
import numpy as np
import statistics
import collections
import matplotlib.pyplot as plt
import codecs
import sys
import pickle
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
train_data_df = pd.read_csv('Data/training_data_original_refine.csv', encoding = "ISO-8859-1")
label=train_data_df
train_data_df= train_data_df.loc[:, (train_data_df != 0).any(axis=0)]
label= label.loc[:, (label != 0).any(axis=0)]
label=label.iloc[:,1:]
label.fillna(0)
label.replace(np.nan, 0, inplace=True)
label.replace(np.inf, 0, inplace=True)
label=label.astype('int32')
#print(label)
train_data_df=train_data_df.iloc[:,0].astype(str)
index=[]
print(np.shape(label))
num_of_sentences=[]
def word_and_sentence_analyze():

    num_of_words=[]
    num_of_words_per_sentence=[]
    for i, sentence in enumerate(train_data_df):
        sentence_list=sent_tokenize(sentence)
        index.append(i)
        num_of_sentences.append(len(sentence_list))
        for j in sentence_list:
            word_list_per_sentence=j.split()
            num_of_words_per_sentence.append(len(word_list_per_sentence))
        word_list=sentence.split()
        num_of_words.append(len(word_list))
    '''
    plt.subplot(2,1,1)
    plt.plot(index,num_of_sentences,color='red')
    plt.title("CTI Report")
    plt.ylabel("Number of Sentences")
    plt.subplot(2,1,2)
    plt.plot(index,num_of_words,color='green')
    plt.xlabel("# of report in dataset")
    plt.ylabel("Number of Words")
    plt.savefig("CTI report.jpg")
    '''
    print("haha",sum(num_of_sentences)/len(num_of_sentences))
    print(min(num_of_sentences),max(num_of_sentences))
    print(min(num_of_words),max(num_of_words))
    print("std= ", statistics.stdev(num_of_sentences))
    print("std= ", statistics.stdev(num_of_words))
    sentence_df=pd.DataFrame(num_of_sentences)
    word_df=pd.DataFrame(num_of_words)
    '''
    myfig=plt.figure()
    plt.subplot(1,2,1)
    plt.title("Bot plot of sentences")
    ax=sentence_df.boxplot()
   
    plt.subplot(1,2,2)
    plt.title("Bot plot of words")
    bx=word_df.boxplot()
    myfig.savefig("QQ.jpg")
    '''
    print(np.percentile(sentence_df,25),np.percentile(sentence_df,50),np.percentile(sentence_df,75))
    print(np.percentile(word_df,25),np.percentile(word_df,50),np.percentile(word_df,75))

#-------------------------------------------- TTP_analysis----------------------------------
TTP_ana=[0]*168
#label=list(label)
print(label)
temp=[]
indexes=[]
def TTP_analyza():
    for i,each_TTP_label in enumerate(train_data_df):
        temp=label.iloc[i,:]
        for j,content in enumerate(temp):
            if(content==1):
                TTP_ana[j]=TTP_ana[j]+1
    for i in range(168):
        indexes.append(i)
    print("lala",sum(TTP_ana)/len(TTP_ana))
    print(min(TTP_ana), max(TTP_ana))
    print("std= ", statistics.stdev(TTP_ana))
    #plt.subplot(2,1,1)
    TTP_df=pd.DataFrame(TTP_ana)
    '''
    plt.plot(indexes,TTP_ana,color='red')
    plt.title("Technique")
    plt.ylabel("Number of Reports")
    plt.savefig("TTP_analysis.jpg")


    myfig=plt.figure()
    plt.title("Box plot of MITRE")
    ax=TTP_df.boxplot()
    myfig.savefig("BOX_TTP.jpg")
    '''
    print(np.percentile(TTP_df,25),np.percentile(TTP_df,50),np.percentile(TTP_df,75))
temp1=[]

num_Tech_per_report=[]
indexeses=[]
def CTI_report_Tech_number():
    avg_TTP=0
    for i,each_TTP_label in enumerate(train_data_df):
        indexeses.append(i)
        temp1=label.iloc[i,:]
        if(i==663):
            print(each_TTP_label)
        avg_TTP=avg_TTP+np.count_nonzero(temp1 == 1)
        num_Tech_per_report.append(np.count_nonzero(temp1 == 1))
    #for jj,jjj in enumerate(num_Tech_per_report):
        #if (jjj==64):
           # print(train_data_df.iloc[jj,0:])
    print(np.shape(num_Tech_per_report))

    plt.plot(indexeses,num_Tech_per_report,color='red')
    plt.title("Technique per report")
    plt.ylabel("Number of Techniques")
    plt.savefig("TTP_per_report.jpg")
    myfig=plt.figure()
    TTP_df=pd.DataFrame(num_Tech_per_report)
    
    plt.title("Box plot of TTP per report")
    ax=TTP_df.boxplot()
    myfig.savefig("BOX_TTP_per_report.jpg")

    print(avg_TTP/695)
    print(min(num_Tech_per_report),max(num_Tech_per_report))
    print("std= ", statistics.stdev(num_Tech_per_report))
    print(np.percentile(TTP_df,25),np.percentile(TTP_df,50),np.percentile(TTP_df,75))
if __name__ == '__main__':
    word_and_sentence_analyze()
    TTP_analyza()
    CTI_report_Tech_number()
    '''
    dictionary={}

    for j,itera in enumerate(num_of_sentences):
        dictionary[itera]=num_Tech_per_report[j]

    dictionary=dictionary.items()
    dictionary=sorted(dictionary)
    print(dictionary)
    x,y=zip(*dictionary)
    plt.plot(x,y, color='red')

    plt.title("CTI Report")
    plt.xlabel("Number of Sentences")
    plt.ylabel("Number of TTPs")
    plt.savefig("Sentence_Tech.jpg")
    '''