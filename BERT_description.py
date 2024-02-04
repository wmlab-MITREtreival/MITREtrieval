import torch

import pandas as pd
import numpy as np
import transformers
import re
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import RobertaModel, RobertaTokenizer
#from torch.utils.data import TensorDataset

def clean_description(df):
    regex = re.compile(r'(?:\([^\(]+\))')
    description_lst=[]

    for i, row in df.iterrows():
        text = row['description']
        if "<code>" in text:
            text = text.replace("<code>", "")
            text = text.replace("</code>", "")
        
        refer = regex.findall(text)
        for r in refer:
            text = text.replace(r, "")
        text = text.replace("  ", " ")
        description_lst.append(text)
    
    df['description'] = description_lst

    return df

MAX_LEN = 512
TRAIN_BATCH_SIZE = 10
VALID_BATCH_SIZE = 2
EPOCHS = 19
TEST_EPOCHS=1
LEARNING_RATE = 5e-5

'''
#df.insert(2,"onehot",y,True)
for i, row in df.iterrows():
    print(y[i])
    temp=np.array(y[i]).reshape(1,-1)
    print(temp.shape)
    temp=temp.tolist()
    df.at[i,'onehot']=temp
'''
#df['onehot']=y


#config.__setattr__('bert_batch_size',self.args['bert_batch_size'])



class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.description
        self.targets = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        text = " ".join(text.split())
        #print(text)
        #if(len(text)>510):
            #print("Over 510")
            #text=text[:255]+text[-255:]
        #print(text)
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }



class BERTClass(torch.nn.Module):
    def __init__(self,config):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased',config=config)
        for param in self.l1.parameters():
            param.requires_grad = True
        self.l2 = torch.nn.Dropout(0.1)
        self.l3 = torch.nn.Linear(768, 188)
        self.emb=768
        #self.sig=torch.nn.Sigmoid()
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)

        output_2=self.l2(output_1)
        #print(np.shape(hidden[0][:,0,:]))
        #output_2=self.l2(hidden[0][:,0,:])
        output = self.l3(output_2)
        #output=self.sig(output)
        #print(output)
        print(np.shape(output_1))
        return output_1,output

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        for param in self.l1.parameters():
            param.requires_grad = True
        
        self.l3 = torch.nn.Linear(768, 188)
        self.l2 = torch.nn.Dropout(0.15)
        self.emb=768
        #self.sig=torch.nn.Sigmoid()
    
    def forward(self, ids, mask, token_type_ids):
        output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        
        output_2=self.l2(output_1[1])
        #print(np.shape(hidden[0][:,0,:]))
        #output_2=self.l2(hidden[0][:,0,:])
        output = self.l3(output_2)
        #output=self.sig(output)
        #print(output)
        #print(np.shape(output_1[0]),np.shape(output_1[1]))
        return output_2,output

def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets.squeeze())

def train(epoch,model):
    model.train()
    i=0
    for _,data in enumerate(training_loader, 0):# get data in a batch
        ids = data['ids'].to(device='cuda:1', dtype = torch.long)
        mask = data['mask'].to(device='cuda:1', dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device='cuda:1', dtype = torch.long)
        targets = data['targets'].to(device='cuda:1', dtype = torch.long)
        pool,outputs = model(ids, mask, token_type_ids)
       
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _%5000==0:
            print('Epoch: ',epoch, 'Loss: ' , loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(),"bert_description.pt")

if __name__=="__main__":
    model_config = BertConfig.from_pretrained('bert-base-uncased')
    model_config.num_attention_Heads=12
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', return_dict=False)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
    df=pd.read_csv('technique_description_f.csv')

    df=clean_description(df)
    possible_labels = df.ID.unique()
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    df['label'] = df.ID.replace(label_dict)

    df = clean_description(df)

    y = LabelBinarizer().fit_transform(df.label).tolist()
    print(df)
    print("TRAIN Dataset: {}".format(df.shape))


    training_set = CustomDataset(df, tokenizer, MAX_LEN)
    #testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
    training_loader = DataLoader(training_set, **train_params)
    model = RobertaClass()
    model.to(device='cuda:1')
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        train(epoch,model)
