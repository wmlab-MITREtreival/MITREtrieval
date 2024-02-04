import pickle
import pandas as pd
from query_list import *
import sys
import codecs
import transformers
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score,classification_report,multilabel_confusion_matrix,fbeta_score
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
class BERTClass(torch.nn.Module):
    def __init__(self,model_config):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased',config=model_config)
        for param in self.l1.parameters():
            param.requires_grad = True
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 2)
        self.tan=torch.nn.Tanh()
    
    def forward(self, ids, mask, token_type_ids):
        #print(mask)
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids,return_dict=False)
        output_2=self.l2(output_1)
        output = self.l3(output_2)
        output=self.tan(output)
        return output
if __name__=="__main__":
   
    with open("binary_bert_cy","rb") as ff:
        dataset=pickle.load(ff)
    ff.close()

    train_data, test_data = train_test_split(dataset, random_state=777, train_size=0.8)
    train_dataset = train_data.reset_index(drop=True)
    test_dataset=test_data.reset_index(drop=True)
    
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 10
    VALID_BATCH_SIZE = 30
    EPOCHS = 10
    TEST_EPOCHS=1
    LEARNING_RATE = 1e-05
    model_config = BertConfig.from_pretrained('bert-base-uncased')
    #model_config.num_attention_Heads=12
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', return_dict=False)
    class CustomDataset(Dataset):

        def __init__(self, dataframe, tokenizer, max_len):
            self.tokenizer = tokenizer
            self.data = dataframe
            self.text = dataframe.Text
            self.targets = self.data.label
            self.max_len = max_len

        def __len__(self):
            return len(self.text)

        def __getitem__(self, index):
            text = str(self.text[index])
            text = " ".join(text.lower().split())

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
                'targets': torch.tensor(self.targets[index], dtype=torch.float)
            }
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)  
    model = BERTClass(model_config)
    model.to('cuda')
    if torch.cuda.device_count() > 1:
        model= torch.nn.DataParallel(model)
    def loss_fn(outputs, targets):
        return torch.nn.CrossEntropyLoss()(outputs, targets)
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)  

    
    
    def train(epoch,model):
        model.train()
        i=0
        for _,data in enumerate(training_loader, 0):# get data in a batch
            ids = data['ids'].to(device='cuda', dtype = torch.long)
            mask = data['mask'].to(device='cuda', dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device='cuda', dtype = torch.long)
            targets = data['targets'].to(device='cuda', dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
        
            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            if _%5000==0:
                print('Epoch: ',epoch, 'Loss: ' , loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(),"bert_binary_cy.pt")
    if sys.argv[1]=='Train':
        for epoch in tqdm(range(EPOCHS)):
            train(epoch,model)
    elif  sys.argv[1]=='Test':
        model.load_state_dict(torch.load("bert_binary_cy.pt"))
        model.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):
                ids = data['ids'].to('cuda', dtype = torch.long)
                mask = data['mask'].to('cuda', dtype = torch.long)
                token_type_ids = data['token_type_ids'].to('cuda', dtype = torch.long)
                targets = data['targets'].to('cuda', dtype = torch.float)
                outputs = model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            #print(fin_outputs)
            #outputs = np.array(fin_outputs) >= 0
            #print(outputs)
            ans=[]
            for it in fin_outputs:
                if it[0]>0:
                    ans.append([0])
                else:
                    ans.append([1])
            print(ans)
            accuracy = metrics.accuracy_score(fin_targets,ans)
            precision=precision_score(fin_targets,ans, pos_label=1)
            recall=recall_score(fin_targets,ans, pos_label=1)
            f1=fbeta_score(fin_targets,ans, pos_label=1, beta=1)
            print(precision,recall,f1)
            print(accuracy)
            recall=recall_score(fin_targets,ans, pos_label=0)
            print("fnr=",1-recall)