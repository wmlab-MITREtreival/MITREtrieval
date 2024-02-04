import pandas as pd
import numpy as np

from nltk import word_tokenize
import re
import pickle
import codecs
import sys
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
'''
 *clean_text()-Clean noise words in text
 * @text: Text in CTI report or in a sentence
 * @return: Clean text
'''
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r'(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})|(?:\d{1,3}\[.]\d{1,3}\[.]\d{1,3}\[.]\d{1,3})', '', text)
    text = re.sub(r'\b(?:CVE\-[0-9]{4}\-[0-9]{4,6})\b', '', text)
    text = re.sub(r'\b(?:[a-z][_a-z0-9-.]+@[a-z0-9-]+\.[a-z]+)\b|\b(?:[a-z][_a-z0-9-[.]]+@[a-z0-9-]+\[.][a-z]+)\b', '', text)
    text = re.sub(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b', '', text)
    text = re.sub(r'\b(?:[a-f0-9]{32}|[A-F0-9]{32})\b', '', text)
    text = re.sub(r'\b(?:(HKLM|HKCU|HKEY|hklm|hkcu|hkey)([\\A-Za-z0-9-_]+))\b', 'registry', text)
    text = re.sub(r'\b(?:[a-f0-9]{40}|[A-F0-9]{40}|[0-9a-f]{40})\b', '', text)
    text = re.sub(r'\b([a-f0-9]{64}|[A-F0-9]{64})\b', '', text)
    text = re.sub(r'(?:[A-Za-z0-9]*\.(?:jpg|JPG|gif|GIF|doc|DOC|pdf|PDF|cpp|c))','',text)
    text = re.sub(r'\b[a-z0-9]+(?:\.[a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj| Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b', '', text)
    text = re.sub(r'\b[a-fA-F\d]{32}\b|\b[a-fA-F\d]{40}\b|\b[a-fA-F\d]{64}\b', '', text)
    text = re.sub(r'x[A-Fa-f0-9]{2}', ' ', text)
    text = re.sub(r'stage \d', '. stage ', text)
    text = re.sub(r'\b(\d{4}-\d{1,2}-\d{1,2})\b', '', text)
    text=text.replace("e ute","execute")
    text=text.replace("e utly","executly")
    text=text.replace("e utable","executable")
    text=text.replace("e uting","executing")
    text=text.replace("e utive","executive")
    text=text.replace("e ution","execution")
    text=text.replace("e cel","excel")
    text = text.strip(' ')
    return text
import copy
def Train_Val_Test_Split_Sentence():
    with open("cycarrier_data","rb") as f:
        dataset=pickle.load(f)
    f.close()
    Technique_num_dict={}
    Technique_name=[]
    for index,sentence in dataset.iterrows():
        if sentence['label'] not in Technique_num_dict:
            
            Technique_num_dict[sentence['label']]=1
        else:
            Technique_num_dict[sentence['label']]=Technique_num_dict[sentence['label']]+1
    temp=copy.deepcopy(Technique_num_dict)
    
    Technique_num_dict = sorted(Technique_num_dict.items(), key=lambda d: d[1]) #Sort by appear times
    Technique_num_dict=dict(Technique_num_dict)
    for key,value in Technique_num_dict.items():
        if value<50:
            del temp[key]
        else:
            Technique_name.append(key)
    Technique_num_dict=temp
    Technique_file=open("Data/Technique_name_sentence.txt","wb")
    pickle.dump(Technique_name,Technique_file)
    print(Technique_num_dict,len(Technique_num_dict))
    Training_set=pd.DataFrame()
    Validation_set=pd.DataFrame()
    Testing_set=pd.DataFrame()
    Training=[]
    Validation=[]
    Testing=[]
    for key,value in Technique_num_dict.items():
        length=dataset[dataset.label==key].shape[0] #get number with soecific 
        Train_num=round(length*0.8)
        Val_num=round(length*0.1)
        Test_num=round(length*0.1)
        temp=dataset[dataset.label==key]
        Training_set= Training_set.append(temp.sample(Train_num))
        temp = temp.loc[temp.index.difference(Training_set.index)]
        Validation_set= Validation_set.append(temp.sample(Val_num))
        temp = temp.loc[temp.index.difference(Validation_set.index)]
        Testing_set= Testing_set.append(temp)
    Training_set= Training_set.reset_index()
    Validation_set= Validation_set.reset_index()
    Testing_set= Testing_set.reset_index()
    train_word=[]
    valid_word=[]
    test_word=[]
    for tr,tr_iter in Training_set.iterrows():
        train_word.append(len(word_tokenize(tr_iter['Text'])))
    for tr,tr_iter in Validation_set.iterrows():
        valid_word.append(len(word_tokenize(tr_iter['Text'])))
    for tr,tr_iter in Testing_set.iterrows():
        test_word.append(len(word_tokenize(tr_iter['Text'])))
    print(Training_set,Validation_set,Testing_set)
    
    print( sum(train_word) / len(train_word),np.std(train_word))
    print( sum(valid_word) / len(valid_word),np.std(valid_word))
    print( sum(test_word) / len(test_word),np.std(test_word))
    return Training_set,Validation_set,Testing_set
def Train_Val_Test_Split_specific_technique(technique_list):
    train_data_df = pd.read_csv('Data/training_data_original_refine1.csv',encoding = "utf-8")
    train_data_df['Text'] = train_data_df['Text'].map(lambda com : clean_text(com))
    train_data_df.drop(train_data_df.columns[[1, 2, 3,4,5,6,7,8,9,10,11,12,13,14]], axis = 1, inplace = True)

    label=train_data_df
    train_data_df= train_data_df.loc[:, (train_data_df != 0).any(axis=0)]
    label= label.loc[:, (label != 0).any(axis=0)]
    label=label.iloc[:,1:]
    label.fillna(0)
    label.replace(np.nan, 0, inplace=True)
    label.replace(np.inf, 0, inplace=True)
    #label=label.astype('int32')
    Technique=list((train_data_df.iloc[0:0,1:].columns))
    train_data_df1=train_data_df.iloc[:,0].astype(str)
    for it,jj in enumerate(Technique): #Delete TTP withless than 3 appear times
        if(Technique[it] not in technique_list):
            train_data_df.drop(Technique[it],axis=1,inplace=True)
    TTP_ana=[0]*len(technique_list)
    for it,jj in train_data_df.iterrows():
        temp=train_data_df.iloc[it,1:]
        for j,content in enumerate(temp):
            if(content==1):
                TTP_ana[j]=TTP_ana[j]+1
    print(TTP_ana,len(TTP_ana),list((train_data_df.iloc[0:0,1:].columns)))
    list1, list2 = zip(*sorted(zip(TTP_ana, list((train_data_df.iloc[0:0,1:].columns)))))
    print(list1,list2)
    with open("Data/Technique_name_"+str(len(list2))+".txt","wb") as f:
        pickle.dump(list((train_data_df.iloc[0:0,1:].columns)),f)
    f.close()
    Total_dataset=train_data_df # ----------Total dataset----------
    Training_set=pd.DataFrame()
    Validation_set=pd.DataFrame()
    Testing_set=pd.DataFrame()
    Training=[]
    Validation=[]
    Testing=[]
    for key in list2:
        #print(len(train_data_df.loc[train_data_df[key] == 1].index))
        length=len(train_data_df.loc[train_data_df[key] == 1].index)
        Train_num=0
        Val_num=0
        Test_num=0
        if length <3:
            continue
        if length<10 :
            Train_num=length-2
            Val_num=1
            Test_num=1
            
            #Validation_set
        else:
            Train_num=round(length*0.8)
            Val_num=round(length*0.1)
            Test_num=round(length*0.1)
            #Training_set.append(train_data_df.loc[train_data_df[key] == 1].sample(round(length*0.8)))
        temp=train_data_df.loc[train_data_df[key] == 1]
        Training_set= Training_set.append(temp.sample(Train_num))
        temp = temp.loc[temp.index.difference(Training_set.index)]
        train_data_df=train_data_df.loc[train_data_df.index.difference(Training_set.index)]
        Validation_set= Validation_set.append(temp.sample(Val_num))
        temp = temp.loc[temp.index.difference(Validation_set.index)]
        train_data_df=train_data_df.loc[train_data_df.index.difference(Validation_set.index)]
        Testing_set= Testing_set.append(temp)
        train_data_df=train_data_df.loc[train_data_df.index.difference(Testing_set.index)]

    print("Tr= ",np.shape(Training_set)," V= ",np.shape(Validation_set)," Te= ",np.shape(Testing_set))
    return Training_set,Validation_set,Testing_set
'''
 * Train_Val_Test_Split()-Split dataset into train(80%)/valid(10%)/test(10%) through support of each technique to make distribution balance
 * @threshold: Technique threshold
 * @return: Training_set,Validation_set,Testing_set
'''
def Train_Val_Test_Split(threshold:str):
    train_data_df = pd.read_csv('Data/training_data_original_refine1.csv',encoding = "utf-8")
    train_data_df['Text'] = train_data_df['Text'].map(lambda com : clean_text(com))
    train_data_df.drop(train_data_df.columns[[1, 2, 3,4,5,6,7,8,9,10,11,12,13,14]], axis = 1, inplace = True)
    label=train_data_df
    train_data_df= train_data_df.loc[:, (train_data_df != 0).any(axis=0)]
    label= label.loc[:, (label != 0).any(axis=0)]
    label=label.iloc[:,1:]
    label.fillna(0)
    label.replace(np.nan, 0, inplace=True)
    label.replace(np.inf, 0, inplace=True)
    Technique=list((train_data_df.iloc[0:0,1:].columns))
    train_data_df1=train_data_df.iloc[:,0].astype(str)
    print(np.shape(Technique))
    TTP_ana=[0]*np.shape(Technique)[0]
    temp=[]
    for i,each_TTP_label in enumerate(train_data_df1):
        temp=label.iloc[i,:]
        for j,content in enumerate(temp):
            if(content==1):
                TTP_ana[j]=TTP_ana[j]+1
    Technique_num_dict={}
    Technique_name=[]
    for it,jj in enumerate(TTP_ana): #Delete TTP withless than 3 appear times
        if(jj<int(threshold)):
            train_data_df.drop(Technique[it],axis=1,inplace=True)
        else:
            Technique_num_dict[Technique[it]]=jj
            Technique_name.append(Technique[it])
    Technique_file=open("Data/Technique_name_"+threshold+".txt","wb")
    pickle.dump(Technique_name,Technique_file)
    Technique_num_dict = sorted(Technique_num_dict.items(), key=lambda d: d[1]) #Sort by appear times
    Technique_num_dict=dict(Technique_num_dict)
    Total_dataset=train_data_df # ----------Total dataset----------
    Training_set=pd.DataFrame()
    Validation_set=pd.DataFrame()
    Testing_set=pd.DataFrame()
    for key,value in Technique_num_dict.items():
        #print(len(train_data_df.loc[train_data_df[key] == 1].index))
        length=len(train_data_df.loc[train_data_df[key] == 1].index)
        Train_num=0
        Val_num=0
        Test_num=0
        if length <3:
            continue
        if length<10 :
            Train_num=length-2
            Val_num=1
            Test_num=1
        else:
            Train_num=round(length*0.8)
            Val_num=round(length*0.1)
            Test_num=round(length*0.1)
            #Training_set.append(train_data_df.loc[train_data_df[key] == 1].sample(round(length*0.8)))
        temp=train_data_df.loc[train_data_df[key] == 1]
        Training_set= Training_set.append(temp.sample(Train_num))
        temp = temp.loc[temp.index.difference(Training_set.index)]
        train_data_df=train_data_df.loc[train_data_df.index.difference(Training_set.index)]
        Validation_set= Validation_set.append(temp.sample(Val_num))
        temp = temp.loc[temp.index.difference(Validation_set.index)]
        train_data_df=train_data_df.loc[train_data_df.index.difference(Validation_set.index)]
        Testing_set= Testing_set.append(temp)
        train_data_df=train_data_df.loc[train_data_df.index.difference(Testing_set.index)]
    print("Tr= ",np.shape(Training_set)," V= ",np.shape(Validation_set)," Te= ",np.shape(Testing_set))
    return Training_set,Validation_set,Testing_set