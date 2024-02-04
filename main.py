from pkg_resources import empty_provider
from transformers import BertTokenizer
import pandas as pd
import torch
import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.metrics import f1_score, precision_score, recall_score,classification_report,multilabel_confusion_matrix,fbeta_score, roc_auc_score,hamming_loss
'''
* @ parameter:model_train :Retrain the DL model
* @ parameter: Inference :Get answer from Ontology
* @ parameter: Fusion:Fused answer from ontology and DL model

'''
def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list
def import_data_fusion(max_len,test_doc,threshold:str):
        try:
                #print("gggggg",threshold)
                with open("sentence_encode_train_"+threshold,"rb") as f:
                        train=pickle.load(f)
    
                with open("sentence_encode_fusion_"+threshold,"rb") as f:
                        data=pickle.load(f)
        except:
                with open('dl_train_feature_'+threshold+'_clean',"rb") as f:
                        train=pickle.load(f)

                #with open('dl_test_feature_'+threshold+'_clean',"rb") as f:
                 #       test=pickle.load(f)
                
                print(train)
                train=encode_doc(train)

    

        train_ = np.zeros((len(train),max_len,768)) 
        test_ = np.zeros((1,max_len,768)) 
        # assigning values
        for idx,doc in enumerate(train): 
                if doc.shape[0]<=max_len:
        #         print(idx)
                        train_[idx][:doc.shape[0],:] = doc
                else:
                        train_[idx][:max_len,:] = doc[:max_len,:]
                
        print('positive example shape: ', train_.shape)

        if test_doc.shape[0]<=max_len:
                test_[0][:test_doc.shape[0],:] = test_doc
        else:
                test_[0][:max_len,:] = test_doc[:max_len,:]
        
        return train_,test_
from nltk import sent_tokenize
def visualization_txt(test):
        sentence_list=[]
        for index,test_iter in test.iterrows():
                sent_list=sent_tokenize(test_iter['Text'])
                if len(sent_list)>=120:
                        sentence_list.append(sent_list[:120])
                else:
                        sent_list=sent_list+['']*(120-len(sent_list))
                        sentence_list.append(sent_list[:120])

        print(np.shape(sentence_list))
        return sentence_list
latex_special_token = ["!@#$%^&*()"]

def generate(sentence_list, attention_list, latex_file, color='red', rescale_value = False):
        assert(len(sentence_list) == len(attention_list))
        if rescale_value:
                attention_list = rescale(attention_list)
        sentence_num = len(sentence_list)
        print(np.shape(attention_list))
        with open(latex_file,'w') as f:
                f.write(r'''\documentclass[varwidth]{standalone}\special{papersize=210mm,297mm}
                \usepackage{color}
                \usepackage{tcolorbox}
                \usepackage{CJK}
                \usepackage{adjustbox}
                \tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
                \begin{document}
                \begin{CJK*}{UTF8}{gbsn}'''+'\n')
                string = r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
                for idx in range(sentence_num):
                        string += "\\colorbox{%s!%s}{"%(color, attention_list[idx])+"\\strut " + sentence_list[idx]+"} "
                string += "\n}}}"
                f.write(string+'\n')
                f.write(r'''\end{CJK*}
                \end{document}''')
import random, os, numpy, scipy
from codecs import open
def createHTML(texts, weights, fileName):
    """
    Creates a html file with text heat.
	weights: attention weights for visualizing
	texts: text on which attention weights are to be visualized
    """
    #fileName = "visualization/"+fileName
    weights=np.array(weights)
    #print(weights)
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
    var color = "255,0,0";
    var ngram_length = 3;
    var half_ngram = 1;
    for (var k=0; k < any_text.length; k++) {
    var tokens = any_text[k];
    var intensity = new Array(tokens.length);
    var max_intensity = Number.MIN_SAFE_INTEGER;
    var min_intensity = Number.MAX_SAFE_INTEGER;
    for (var i = 0; i < intensity.length; i++) {
    intensity[i] = 0.0;
    for (var j = -half_ngram; j < ngram_length-half_ngram; j++) {
    if (i+j < intensity.length && i+j > -1) {
    intensity[i] += trigram_weights[k][i + j];
    }
    }
    if (i == 0 || i == intensity.length-1) {
    intensity[i] /= 2.0;
    } else {
    intensity[i] /= 3.0;
    }
    if (intensity[i] > max_intensity) {
    max_intensity = intensity[i];
    }
    if (intensity[i] < min_intensity) {
    min_intensity = intensity[i];
    }
    }
    var denominator = max_intensity - min_intensity;
    for (var i = 0; i < intensity.length; i++) {
    intensity[i] = (intensity[i] - min_intensity) / denominator;
    }
    if (k%2 == 0) {
    var heat_text = "<p><br><b>CTI report Example:</b><br>";
    } else {
    var heat_text = "<b>CTI report Example:</b><br>";
    }
    var space = "";
    for (var i = 0; i < tokens.length; i++) {
    heat_text += "<span style='background-color:rgba(" + color + "," + intensity[i] + ")'>"  + space + tokens[i] + "</span>" + "<br>";
    if (space == "") {
    space = " ";
    }
    }
    //heat_text += "<p>";
    document.body.innerHTML += heat_text;
    }
    </script>
    </html>"""
    putQuote = lambda x: "\"%s\""%x
    textsString = "var any_text = [%s];\n"%(texts)
    weightsString = "var trigram_weights = [[%s]];\n"%(",".join(map(str,weights)))
    fOut.write(part1)
    fOut.write(textsString)
    fOut.write(weightsString)
    fOut.write(part2)
    fOut.close()
  
    return
if __name__=="__main__":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--stage', type=str, required=True)
        parser.add_argument('--threshold', type=str, required=True)
        args = parser.parse_args()
        Technique_name_f=open("Data/Technique_name_"+args.threshold+".txt","rb")
        Technique_name=pickle.load(Technique_name_f)
        Technique_name_f.close()
        if args.stage=='Inference':
                from attack_pattern_inference import *
                from tqdm import tqdm
                f=open('ontology_feature_'+args.threshold,'rb')          
                ontology_feature=pickle.load(f)
                f.close()
                
                '''
                ontology_feature:
                Column: Text, Correct_output, v_o,Group,Soft
                '''
                
                ontology_feature_clean=[]
                ans_from_ontology=[]

                for ind,feature_v in ontology_feature.iterrows():
                        for indexeses,feature_v_o in enumerate(feature_v['v_o']):
                                if np.shape(feature_v_o)[0]>=2:
                                #print(feature_v_o)
                                        feature_v['v_o'][indexeses]=feature_v['v_o'][indexeses][:2]
                                if np.shape(feature_v_o)[0]<2:
                                        #print(feature_v_o)
                                        feature_v['v_o'][indexeses].append(" ")
                        print("shpae",np.shape(feature_v['v_o']))
                        ontology_feature_clean.append(feature_v)
                
                ontology_feature_clean=pd.DataFrame(ontology_feature_clean,columns=['Text','list','Group','Soft','v_o','srl'])
                f=open('ontology_feature_'+args.threshold,'wb')
                pickle.dump(ontology_feature_clean,f)
                ontology_embeddings = get_embedding()
                print(ontology_feature_clean)
                for index,feature in tqdm(ontology_feature_clean.iterrows()):
                        print(index)
                        ans_from_ontology.append(attack_pattern_inference(ontology_embeddings,feature['Group'],feature['Soft'],feature['v_o'],feature['Text'],feature['srl']))
                        #ans_from_ontology.append(attack_pattern_inference(feature['Group'],feature['Soft'],feature['v_o']))

                        if index % 10==0:
                                with open ("all_answer_from_ontology_"+args.threshold,"wb") as f:
                                        pickle.dump(ans_from_ontology,f)
                print(ans_from_ontology)
                with open ("all_answer_from_ontology_"+args.threshold,"wb") as f:
                        pickle.dump(ans_from_ontology,f)
        
        elif args.stage=='Inference_train':
                from attack_pattern_inference import *
                from tqdm import tqdm
                f=open('ontology_feature_'+args.threshold+'train','rb')          
                ontology_feature=pickle.load(f)
                f.close()
                
                '''
                ontology_feature:
                Column: Text, Correct_output, v_o,Group,Soft
                '''
                
                ontology_feature_clean=[]
                ans_from_ontology=[]

                for ind,feature_v in ontology_feature.iterrows():
                        for indexeses,feature_v_o in enumerate(feature_v['v_o']):
                                if np.shape(feature_v_o)[0]>=2:
                                #print(feature_v_o)
                                        feature_v['v_o'][indexeses]=feature_v['v_o'][indexeses][:2]
                                if np.shape(feature_v_o)[0]<2:
                                        #print(feature_v_o)
                                        feature_v['v_o'][indexeses].append(" ")
                        ontology_feature_clean.append(feature_v)
                
                ontology_feature_clean=pd.DataFrame(ontology_feature_clean,columns=['Text','list','Group','Soft','v_o','srl'])
                f=open('ontology_feature_'+args.threshold+'train','wb')
                pickle.dump(ontology_feature_clean,f)
                vo_pair_embeddings = get_embedding()
                print(ontology_feature_clean)
                for index,feature in tqdm(ontology_feature_clean.iterrows()):
                        print(index)
                        ans_from_ontology.append(attack_pattern_inference(vo_pair_embeddings, feature['Group'],feature['Soft'],feature['v_o'],feature['Text'],feature['srl']))
                        if index % 10==0:
                                with open ("all_answer_from_ontology_"+args.threshold+'train',"wb") as f:
                                        pickle.dump(ans_from_ontology,f)
                print(ans_from_ontology)
                with open ("all_answer_from_ontology_"+args.threshold+'train',"wb") as f:
                        pickle.dump(ans_from_ontology,f)
        elif args.stage=='encode_sen_in_doc':
                from model.encode_document import encode_sen_in_doc
                with open('dl_test_feature_'+args.threshold+'_clean','rb') as f:
                        train=pickle.load(f)
                emb=encode_sen_in_doc(train)
                with open('doc_in_sen_emb_'+args.threshold,'wb') as f:
                        pickle.dump(emb,f)
        elif args.stage=='train_test_split_sentence':
                from model.Train_Test_split import Train_Val_Test_Split_Sentence
                
                train,val,test=Train_Val_Test_Split_Sentence()
                f=open('sentence_train','wb')
                pickle.dump(train,f)
                f.close()
                f=open('sentence_valid','wb')
                pickle.dump(val,f)
                f.close()
                f2=open('sentence_test','wb')
                pickle.dump(test,f2)
                f2.close()
        elif args.stage=='train_test_split_range':
                from model.Train_Test_split import Train_Val_Test_Split_specific_technique
                ## Split Train Test
                train_,val_,test_=Train_Val_Test_Split_specific_technique(Technique_name)
                
                train_=[train_,val_]
                train_=pd.concat(train_)
                
                train_['list'] = train_[train_.columns[1:]].values.tolist()
                train_ = train_[['Text', 'list']].copy()
                test_['list'] = test_[test_.columns[1:]].values.tolist()
                test_ = test_[['Text', 'list']].copy()
                train_ = train_.reset_index(drop=True)
                test_=test_.reset_index(drop=True)
                test_['list']=list(test_['list'])


                f=open('dl_train_feature_'+str(len(Technique_name)),'wb')
                pickle.dump(train_,f)
                f.close()
                f2=open('dl_test_feature_'+str(len(Technique_name)),'wb')
                pickle.dump(test_,f2)
                f2.close()
                print("Train= ",np.shape(train_),"Test= ",np.shape(test_))
                print("===========================================Finish Train Test Split===========================================")
        elif args.stage=='train_test_split':
                from model.Train_Test_split import Train_Val_Test_Split

                ## Split Train Test
                train_,val_,test_=Train_Val_Test_Split(args.threshold)
                
                train_=[train_,val_]
                train_=pd.concat(train_)
                
                train_['list'] = train_[train_.columns[1:]].values.tolist()
                train_ = train_[['Text', 'list']].copy()
                test_['list'] = test_[test_.columns[1:]].values.tolist()
                test_ = test_[['Text', 'list']].copy()
                train_ = train_.reset_index(drop=True)
                test_=test_.reset_index(drop=True)
                print(test_['Text'])
                test_['list']=list(test_['list'])


                f=open('dl_train_feature_'+args.threshold,'wb')
                pickle.dump(train_,f)
                f.close()
                f2=open('dl_test_feature_'+args.threshold,'wb')
                pickle.dump(test_,f2)
                f2.close()
                print("Train= ",np.shape(train_),"Test= ",np.shape(test_))
                print("===========================================Finish Train Test Split===========================================")

        elif args.stage=='preprocess':
                from filter_bert_usage import Preprocess
                print("===========================================Start Preprocessing===========================================")
                Preprocess(args.threshold)
                print("===========================================Finish Preprocessing===========================================")
        elif args.stage=='feature_extraction': 
                from Preprocess.preprocess import Feature_extract_ontology
                from model.encode_document import encode_doc
                with open('dl_train_feature_'+args.threshold+'_clean',"rb") as f:
                        train_=pickle.load(f)
                with open("dl_test_feature_"+args.threshold+"_clean","rb") as of:
                        test_=pickle.load(of)
                
                print("===========================================Start extract ontology feature Train===========================================")
                ontology_feature=Feature_extract_ontology(train_,args.threshold,'train')
                print("===========================================Start extract ontology feature Test===========================================")
                ontology_feature=Feature_extract_ontology(test_,args.threshold)
                print("===========================================Finish extract ontology feature===========================================")
        elif args.stage=='Encode':
                from model.encode_document import encode_doc
                with open('dl_train_feature_'+args.threshold+'_clean',"rb") as f:
                        train_=pickle.load(f)
                with open("dl_test_feature_"+args.threshold+'_clean',"rb") as of:
                        test_=pickle.load(of)
                rep=encode_doc(train_)
                rep1=encode_doc(test_)
                with open("sentence_encode_train_"+args.threshold,"wb") as f:
                        pickle.dump(rep,f)
                
                with open("sentence_encode_test_"+args.threshold,"wb") as f:
                        pickle.dump(rep1,f)
        elif args.stage=='Encode_sen':
                from model.encode_document import encode_doc,encode_sen
                with open('sentence_train',"rb") as f:
                        train_=pickle.load(f)
                with open('sentence_valid',"rb") as of:
                        valid_=pickle.load(of)
                with open('sentence_test',"rb") as of:
                        test_=pickle.load(of)
                rep=encode_sen(train_)
                rep1=encode_sen(valid_)
                rep2=encode_sen(test_)
                with open("sentence_encode_train","wb") as f:
                        pickle.dump(rep,f)
                with open("sentence_encode_valid","wb") as f:
                        pickle.dump(rep1,f)
                with open("sentence_encode_test","wb") as f:
                        pickle.dump(rep2,f)
        elif args.stage=='model_train_sen':
                from model.sentence_bert import sentence_model_train
                sentence_model_train(args.threshold)
        elif args.stage=='model_fusion':
                from model.sentence_bert import MITREtrieval
                MITREtrieval(args.threshold)
        elif args.stage=='model_train_one':
                from model.sentence_bert import one_sentence_model_train
                one_sentence_model_train(args.threshold)
        elif args.stage=='MITREtreival':
                from filter_bert_usage import topic_classifier
                import tqdm                
                from Preprocess.preprocess import query_node_extract
                from model.encode_document import encode_sentence_doc,encode_doc,encode_description_finetuned
                from attack_pattern_inference import *
                from model.sentence_bert import init_weights,BertConfig,Normalize,MITREtrievalmodel,import_ontology_answer

                f=open('input.txt','r', encoding="utf-8")
                doc=f.read()
                print(doc)
                #doc="Quantum Software:  LNK File-Based Builders Growing In Popularity. Possibly Associated With Lazarus APT Group Cyble Research Labs has constantly been tracking emerging threats and their delivery mechanisms. We have observed a surge in the use of .lnk files by various malware families. Some of the prevalent malware families using .lnk files for their payload delivery of late are: Emotet Bumblebee Qbot Icedid Additionally, we have seen many APT instances where the Threat Actors (TAs) leverage .lnk files for their initial execution to deliver the payload. .lnk files are shortcut files that reference other files, folders, or applications to open them. The TAs leverages the .lnk files and drops malicious payloads using LOLBins. LOLBins (Living off the Land Binaries) are binaries that are native to Operating Systems such as PowerShell and mshta. TAs can use these types of binaries to evade detection mechanisms as these binaries are trusted by Operating Systems. During our OSINT (Open Source Intelligence) activity, Cyble Research Labs came across a new. lnk builder dubbed “Quantum Software/Quantum Builder.” Figure 1 shows a post made by the Threat Actor on a cybercrime forum. Figure 1 – Post made by TA on a cybercrime forum The TA claims that Quantum Builder can spoof any extension and has over 300 different icons available for malicious .lnk files. Figure 2 shows the pricing details and functionality of the builder. Figure 2 – Functionality and pricing details The TA has created a video demonstrating how to build .lnk, .hta, and .iso files using the Quantum Builder. The .hta payload can be created using Quantum Builder by customizing options such as payload URL details, DLL support, UAC Bypass, execution path and time delay to execute the payload, etc. Figure 3 – .hta builder The .lnk builder embeds the generated .hta payload and creates a new .lnk file. The builder provides various icons as an option while building the .lnk file. The below figure shows the Quantum .lnk builder. Figure 4 – .lnk builder At the end of this process, the .iso builder is used to create the .iso image containing the .lnk file for further delivery via email and execution. Figure 5 – .iso builder The TA has also claimed to have implemented a dogwalk n-day exploit.  This vulnerability exists in Microsoft Support Diagnostic Tool (MSDT) and could lead to code execution if the user opens a specially crafted .diagcab file, typically sent over emails by TAs. The .diagcab file further downloads a malicious file into the startup folder, which will be executed every time the user logs in. Figure 6 – DogWalk implementation Technical Analysis Further investigation revealed a post shared by the TA, indicating that this sample might be generated using Quantum Builder. (SHA256: 2f6c1def83936139425edfd611a5a1fbaa78dfd3997efec039f9fd3338360d25). The figure below shows the post made by the TA regarding the above sample. Figure 7 – Twitter post linked by TA on a cybercrime forum The sample mentioned in the above post connects to a domain named “quantum-software.online”; the same domain was used by quantum TA as a demo site, as mentioned in the figure below. This indicates that the identified hash is generated using the quantum builder. Figure 8 – Demo site used by TA This sample is a Windows Shortcut (.LNK) file. By default, Windows hides the .lnk extension, so if a file is named as file_name.txt.lnk, then only file_name.txt will be visible to the user even if the show file extension optionis enabled. For such reasons, this might be an attractive option for TAs, using the .lnk files as a disguise or smokescreen. Figure 9 – File details Upon execution, the .Ink file runs the malicious PowerShell code, which executes a .hta file hosted in the remote site using mshta. This script uses a function that deobfuscates the malicious PowerShell script. The function performs a mathematical operation that converts a numeric value into characters. The figure below shows the deobfuscated data. Figure 10 – De-obfuscated data Command: “C:\Windows\system32\mshta.exe” hxxps[:]//quantum-software[.]online/remote/bdg[.]hta The infection chain is represented below. Figure 11 – Infection Chain Possible Links To Lazarus APT In recent samples and research conducted on Lazarus APT, we observed that TAs were using .Lnk for delivering further stage payloads. Upon comparing both scripts, we found that the deobfuscation loop and initialization of variables were the same, indicating the possibility of a connection between Quantum Builder and Lazarus APT group. Figure 12 – Similar PowerShell script Conclusion We have observed a steadily increasing number of high-profile TAs shifting back to .lnk files to deliver their payloads. Typically, TAs use LOLBins in such infection mechanisms because it makes detecting malicious activity significantly harder. The MSDT zero-day vulnerability, which researchers recently discovered, was also exploiting a LOLBin. Within a short window from this incident being observed in the wild, TAs have leveraged this vulnerability using different attack vectors."
                print("===========================================Start Preprocessing===========================================")
                doc=topic_classifier(doc)
                print("===========================================Finish Preprocessing===========================================")
                print("===========================================Start Extracting Query nodes===========================================")
                group,software,all_node,v_o=query_node_extract(doc)
                print("===========================================Finish Extracting Query nodes===========================================")
                print("===========================================Start Query COMAT===========================================")
                for indexeses,feature_v_o in enumerate(v_o):
                        if np.shape(feature_v_o)[0]>=2:
                                v_o[indexeses]=v_o[indexeses][:2]
                        if np.shape(feature_v_o)[0]<2:
                                v_o[indexeses].append(" ")
                vo_pair_embeddings = get_embedding()
                COMAT_result=attack_pattern_inference(vo_pair_embeddings,group,software,v_o,doc,all_node)
                l1_ans_cor=COMAT_result[0]+COMAT_result[1]+COMAT_result[2]
                l2_ans_cor=COMAT_result[3]+COMAT_result[4]
                l3_ans_cor=COMAT_result[5]
                must_ttp=COMAT_result[6]
                must_not_ttp=COMAT_result[7]
                temp1=torch.zeros(( len(Technique_name)))
                temp2=torch.zeros(( len(Technique_name)))
                temp3=torch.zeros(( len(Technique_name)))
                temp4=torch.zeros(( len(Technique_name)))
                temp5=torch.zeros(( len(Technique_name)))
                for l1_ans in l1_ans_cor: 
                        if l1_ans.upper() in Technique_name:
                                temp1[Technique_name.index(l1_ans.upper())]=1
                for l2_ans in l2_ans_cor:
                        if l2_ans.upper() in Technique_name:
                                #print(l2_ans.upper(),Technique_name.index(l2_ans.upper()))
                                temp2[Technique_name.index(l2_ans.upper())]=1
                for l3_ans in l3_ans_cor :
                        if l3_ans.upper() in Technique_name:
                                #print(l3_ans.upper(),Technique_name.index(l3_ans.upper()))
                                temp3[Technique_name.index(l3_ans.upper())]=1
                for l4_ans in must_ttp :
                        if l4_ans.upper() in Technique_name:
                                #print("must=",l4_ans.upper(),Technique_name.index(l4_ans.upper()))
                                temp4[Technique_name.index(l4_ans.upper())]=1
                for l5_ans in must_not_ttp :
                        if l5_ans.upper() in Technique_name:
                                # print("must=",l5_ans.upper(),Technique_name.index(l4_ans.upper()))
                                temp5[Technique_name.index(l5_ans.upper())]=1
                
                print("===========================================Finish Query COMAT===========================================")
                print("===========================================Start Encode CTI report===========================================")
                max_len=125
                temp1=torch.unsqueeze(temp1, 0)
                temp2=torch.unsqueeze(temp2, 0)
                temp3=torch.unsqueeze(temp3, 0)
                temp4=torch.unsqueeze(temp4, 0)
                temp5=torch.unsqueeze(temp5, 0)

                embedding=encode_sentence_doc(doc)
                embedding=np.array(embedding)
                train_emb,embedding=import_data_fusion(max_len,embedding,args.threshold)
                print(embedding.shape)
                print("===========================================Finish Encode CTI report===========================================")
                print("===========================================Start Predicting===========================================")
                config = BertConfig(num_labels=len(Technique_name), num_attention_heads=12)
                des_df=pd.read_csv('technique_description_f.csv')
                des_emb_list=encode_description_finetuned(des_df,Technique_name)
                des_emb_list=torch.FloatTensor(des_emb_list).cuda()
                model =MITREtrievalmodel(config,des_emb_list)
                model.apply(init_weights)
                model.cuda()
                x_train=train_emb
                x_test=embedding
                normalizer = Normalize()
                x_train, x_test = normalizer.normalize(x_train, x_test,max_len) 
                tensor_val_x = torch.from_numpy(x_test).type(torch.FloatTensor)
                tensor_val_y= torch.zeros((1,np.shape(Technique_name)[0])).type(torch.FloatTensor)
                val_set = torch.utils.data.TensorDataset(tensor_val_x,temp1,temp2,temp3,temp4,temp5)
                testloader=torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)
                model.load_state_dict(torch.load("sentence_BERT_model_"+args.threshold+".pt"))
                model.train(False)
                with torch.no_grad():
                        for data in testloader:
                                inputs, l1,l2,l3,must,must_not  = data
                                if inputs.size(1) > config.seq_length:
                                        inputs = inputs[:, :config.seq_length, :]
                                if torch.cuda.is_available():
                                        # inputs, labels = Variable(inputs.to('cuda')), labels.to('cuda')
                                        inputs = inputs.cuda()
                                out = model(inputs,l1.cuda(),l2.cuda(),l3.cuda(),must.cuda(),must_not.cuda())
 

                        print("MITRE ATT&CK Technique= ")
                        ans=out[0].tolist()
                        for id,tech in enumerate(Technique_name):
                                if(ans[0][id]>0):
                                        print(Technique_name[id])
                print("===========================================Finish Predicting===========================================")
        elif args.stage=='visualization':
                from tqdm import tqdm
                import torch
                from torch.autograd import Variable
                from model.sentence_bert import HTransformer2,HTransformer,init_weights,BertConfig,Normalize,MITREtrievalmodel,import_ontology_answer
                from model.encode_document import encode_description,encode_procedure,encode_doc,encode_description_finetuned  
                Technique_name_f=open("Data/Technique_name_"+args.threshold+".txt","rb")
                Technique_name=pickle.load(Technique_name_f)
                Technique_name_f.close()
                max_len=125
                config = BertConfig(num_labels=len(Technique_name), num_attention_heads=12)
                des_df=pd.read_csv('technique_description_f.csv')
                des_emb_list=encode_description_finetuned(des_df,Technique_name)
                des_emb_list=torch.FloatTensor(des_emb_list).cuda()
                train_emb,test_emb=import_data_fusion(max_len,args.threshold)
                ontology_answer_l1_t,ontology_answer_l2_t,ontology_answer_l3_t,must_ttp_ans_t,must_not_ttp_ans_t=import_ontology_answer(args.threshold,Technique_name)
                model =MITREtrievalmodel(config,des_emb_list)
                model.apply(init_weights)
                model.cuda()
                with open('dl_train_feature_'+args.threshold+'_clean',"rb") as f:
                        train_=pickle.load(f)
                with open("dl_test_feature_"+args.threshold+'_clean',"rb") as f:
                        test_=pickle.load(f)
                
                ontology_feature=test_
                x_train=train_emb
                y_train=train_['list']
                x_test=test_emb
                y_test=test_['list']
                sentence_list=visualization_txt(test_)
                
                normalizer = Normalize()
                print("su ",np.shape(x_train))
                x_train, x_test = normalizer.normalize(x_train, x_test,max_len) 
                print(np.shape(y_test))
                correct_output = torch.FloatTensor(y_test)
                tensor_val_x = torch.from_numpy(x_test).type(torch.FloatTensor)
                tensor_val_y = torch.FloatTensor(y_test).type(torch.FloatTensor)
                print(np.shape(tensor_val_x),np.shape(tensor_val_y))
                val_set = torch.utils.data.TensorDataset(tensor_val_x,tensor_val_y,ontology_answer_l1_t,ontology_answer_l2_t,ontology_answer_l3_t,must_ttp_ans_t,must_not_ttp_ans_t)
                testloader=torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)
                model.load_state_dict(torch.load("sentence_BERT_model_"+args.threshold+".pt"))
                model.train(False)
                with torch.no_grad():
                        predictions = torch.empty((tensor_val_y.shape[0], len(Technique_name)))
                        attention_scores = torch.Tensor()
                        for idx, data in enumerate(tqdm(testloader)):
                                inputs, labels, l1,l2,l3,must,must_not  = data
                                if inputs.size(1) > config.seq_length:
                                        inputs = inputs[:, :config.seq_length, :]
                                if torch.cuda.is_available():
                                        # inputs, labels = Variable(inputs.to('cuda')), labels.to('cuda')
                                        inputs, labels = Variable(inputs.cuda()), labels.cuda()



                                out = model(inputs,l1.cuda(),l2.cuda(),l3.cuda(),must.cuda(),must_not.cuda())
                    
                    
                                pred_prob = out[0].cpu()
                                #print(out[1])
                                predictions[idx]=pred_prob
                                last_layer_attention = out[1][-1].cpu()
                                attention_scores = torch.cat((attention_scores, last_layer_attention))
                                last_layer_attention=last_layer_attention.squeeze(0)
                                
                                #generate(sentence_list[idx],last_layer_attention[0][:][0],"sample.tex",'red')
                                temp_att=[0]*max_len
                                
                                for last_iter in last_layer_attention[4]:
                                    for id,last_iter_iter in enumerate(last_iter):
                                            temp_att[id]=temp_att[id]+last_iter_iter    
                                print(temp_att)
                                
                                createHTML(sentence_list[idx],temp_att,"sample_"+str(idx)+".html")
                                del inputs, labels, out
