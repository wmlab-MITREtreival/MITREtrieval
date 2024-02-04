
import joblib
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.base import BaseEstimator, TransformerMixin
import re
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, classification_report,multilabel_confusion_matrix, recall_score,fbeta_score
from nltk import word_tokenize		  
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
import nltk
import pickle
import codecs
import sys
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
nltk.download('stopwords')
nltk.download('omw-1.4')
TEXT_FEATURES = ["processed"]

def createHTML(texts, weights, fileName):
    """
    Creates a html file with text heat.
	weights: attention weights for visualizing
	texts: text on which attention weights are to be visualized
    """
    #fileName = "visualization/"+fileName

    print(np.shape(weights),np.shape(word_tokenize(texts[0])))
    texts[0]=word_tokenize(texts[0])
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
    heat_text += "<span style='background-color:rgba(" + color + "," + intensity[i] + ")'>"  + space + tokens[i] + "</span>";
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
    textsString = "var any_text = %s;\n"%(texts)
    weightsString = "var trigram_weights = [[%s]];\n"%(",".join(map(str,weights)))
    fOut.write(part1)
    fOut.write(textsString)
    fOut.write(weightsString)
    fOut.write(part2)
    fOut.close()
  
    return
class StemTokenizer(object):
	"""
	Transform each word to its stemmed version
	e.g. studies --> studi
	"""
	def __init__(self):
		self.st = EnglishStemmer()
		
	def __call__(self, doc):
		return [self.st.stem(t) for t in word_tokenize(doc)]

class LemmaTokenizer(object):
	"""
	Transform each word to its lemmatized version
	e.g. studies --> study
	"""
	def __init__(self):
		self.wnl = WordNetLemmatizer()
		
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
class TextSelector(BaseEstimator, TransformerMixin):
	"""
	Transformer to select a single column from the data frame to perform additional transformations on
	Use on text columns in the data
	"""
	def __init__(self, key):
		self.key = key

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return X[self.key]
def processing(df):
	"""
	Creating a function to encapsulate preprocessing, to make it easy to replicate on submission data
	"""
	df['Text'] = df['Text'].map(lambda com : clean_text(com))
	return(df)

if __name__=="__main__":
    
    #label=label.astype('int32')
    stop_words = stopwords.words('english')
    new_stop_words = ["'ll", "'re", "'ve", 'ha', 'wa',"'d", "'s", 'abov', 'ani', 'becaus', 'befor', 'could', 'doe', 'dure', 'might', 'must', "n't", 'need', 'onc', 'onli', 'ourselv', 'sha', 'themselv', 'veri', 'whi', 'wo', 'would', 'yourselv']
    stop_words.extend(new_stop_words)

    '''
    reports = train_data_df[TEXT_FEATURES]
    train_reports=reports.iloc[0:500]
    tactics = label.iloc[0:500,0:14]
    techniques = label.iloc[0:500,14:]
    test_report=reports.iloc[500:]
    test_tactics=label.iloc[500:,0:14]
    test_tech=label.iloc[500:,14:]
    '''

    
    with open('dl_train_feature_6',"rb") as f:
        train_=pickle.load(f)
    with open("dl_test_feature_6","rb") as of:
        valid_=pickle.load(of)
    
    '''
    with open('sentence_train',"rb") as f:
            train_=pickle.load(f)
    with open('sentence_valid',"rb") as of:
            valid_=pickle.load(of)

    f.close()
    of.close()
    

    possible_labels = train_.label.unique()
    possible_labels_val=valid_.label.unique()

    #print(possible_labels)
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    train_['label'] = train_.label.replace(label_dict)
    label_dict_val = {}
    for index, possible_label in enumerate(possible_labels_val):
        label_dict_val[possible_label] = index
    valid_['label'] = valid_.label.replace(label_dict_val)
    '''
    tfidf=TfidfVectorizer(tokenizer =StemTokenizer(), stop_words = stop_words, min_df = 2, max_df = 0.99)
    tfidf.fit(train_['Text'])
    

    feature=tfidf.get_feature_names()
    pipeline_techniques = Pipeline([
            ('columnselector', TextSelector(key = 'Text')),
            ('tfidf', TfidfVectorizer(tokenizer =StemTokenizer(), stop_words = stop_words, min_df = 2, max_df = 0.99)),
            ('selection', SelectPercentile(chi2, percentile = 50)),
            ('classifier', OneVsRestClassifier(LinearSVC(penalty = 'l2', loss = 'squared_hinge',dual = False, max_iter = 1000, class_weight = 'balanced'), n_jobs = 1))
        ])
    # train the model for techniques
    pipeline_techniques.fit(train_, list(train_['list']))
    #pipeline_techniques.fit(train_, train_['label'])
    ans=pipeline_techniques.predict(valid_)
    print(np.count_nonzero(ans==0))
    #Technique_name_f=open("Data/Technique_name_sentence.txt","rb")
    Technique_name_f=open("Data/Technique_name_6.txt","rb")
    Technique_name=pickle.load(Technique_name_f)
    Technique_name_f.close()

    #print(np.shape(ans),np.shape(test_tech))
    #confusion=p.DataFrame(multilabel_confusion_matrix(y_true = list(valid_['list']), y_pred = ans).reshape(np.shape(ans)[1],2*2))
    #confusion.to_csv('rcatt_conf.csv',index=True)
    print(ans)
    #cm=confusion_matrix(y_true = valid_['label'], y_pred = ans)
    #print(cm,np.shape(cm))
    cm=multilabel_confusion_matrix(y_true = list(valid_['list']), y_pred = ans).reshape(np.shape(ans)[1],2*2)
    print(cm,np.shape(cm))
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
    print(fpr)
    print(fp,fp/(tn+fp),fn/(tp+fn),sum(fpr)/len(fpr),sum(fnr)/len(fnr))
    print(fbeta_score(y_true =  list(valid_['list']), y_pred = ans, average='micro', beta=2),fbeta_score(y_true =  list(valid_['list']), y_pred = ans, average='macro', beta=2))
    clsf_report = pd.DataFrame(classification_report(y_true =  list(valid_['list']), y_pred = ans, output_dict=True,target_names=Technique_name)).transpose()
    clsf_report.to_csv('rcATT_result2_6.csv')
    
    test_report=["New Banking Malware Uses Network Sniffing for Data Theft New Banking Malware Uses Network Sniffing for Data Theft Posted on:June 27, 2014 at 8:47 am Posted in:Malware, Spam Author: Joie Salvio (Threat Response Engineer) 0 With online banking becoming routine for most users, it comes as no surprise that we are seeing more banking malware enter the threat landscape. In fact, 2013 saw almost a million new banking malware variantsdouble the volume of the previous year. The rise of banking malware continued into this year, with new malware and even new techniques. Just weeks after we came across banking malware that abuses a Window security feature, we have also spotted yet another banking malware. What makes this malware, detected as EMOTET, highly notable is that it sniffs network activity to steal information. The Spam Connection EMOTET variants arrive via spammed messages. These messages often deal with bank transfers and shipping invoices. Users who receive these emails might be persuaded to click the provided links, considering that the emails refer to financial transactions. Figure 1. Sample spammed message Figure 2. Sample spammed message The provided links ultimately lead to the downloading of EMOTET variants into the system. Theft via Network Sniffing Once in the system, the malware downloads its component files, including a configuration file that contains information about banks targeted by the malware. Variants analyzed by engineers show that certain banks from Germany were included in the list of monitored websites. Note, however, that the configuration file may vary. As such, information on the monitored banks may also differ depending on the configuration file. Another downloaded file is a .DLL file that is also injected to all processes and is responsible for intercepting and logging outgoing network traffic. When injected to a browser, this malicious DLL compares the accessed site with the strings contained in the previously downloaded configuration file. If strings match, the malware assembles the information by getting the URL accessed and the data sent. The malware saves the whole content of the website, meaning that any data can be stolen and saved. EMOTET can even sniff out data sent over secured connections through its capability to hook to the following Network APIs to monitor network traffic: PR_OpenTcpSocket PR_Write PR_Close PR_GetNameForIndentity Closesocket Connect Send WsaSend Our researchers attempts to log in were captured by the malware, despite the sites use of HTTPS. Figures 3 and 4. Login attempt captured by the malware This method of information theft is notable as other banking malware often rely on form field insertion or phishing pages to steal information. The use of network sniffing also makes it harder to users to detect any suspicious activity as no changes are visibly seen (such as an additional form field or a phishing page). Moreover, it can bypass even a supposedly secure connection like HTTPs which poses dangers to the users personal identifiable information and banking credentials. Users can go about with their online banking without every realizing that information is being stolen. The Use of Registry Entries Registry entries play a significant role in EMOTETs routines. The downloaded component files are placed in separate entries. The stolen information is also placed in a registry entry after being encrypted. The decision to storing files and data in registry entries could be seen as a method of evasion. Regular users often do not check registry entries for possibly malicious or suspicious activity, compared to checking for new or unusual files. It can also serve as a countermeasure against file-based AV detection for that same reason. Were currently investigating how this malware family sends the gathered data it sniff from the network. Exercising Caution Latest feedback from the Smart Protection Network shows that EMOTET infections are largely centered in the EMEA region, with Germany as the top affected country. This isnt exactly a surprise considering that the targeted banks are all German. However, other regions like APAC and North America have also seen EMOTET infections, implying that this infection is not exclusive to a specific region or country. As EMOTET arrives via spammed messages, users are advised not to click links or download files that are unverified. For matters concerning finances, its best to call the financial or banking institution involved to confirm the message before proceeding. Trend Micro blocks all related threats. With additional insights from Rhena Inocencio and Marilyn Melliang. Update as of July 3, 2014, 2:00 A.M. PDT: The SHA1 hash of the file with this behavior weve seen is: ba4d56d01fa5f892bc7542da713f241a46cfde85 "]
    stem=StemTokenizer()
    doc=stem(test_report[0])
    x=tfidf.transform(test_report)

    f_list=list(x.toarray())
    #print(np.shape(f_list))

    #temp=doc.split(" ")
    
    feature_list=[0]*np.shape(doc)[0]
    feature_name=tfidf.get_feature_names()

    doc_cor=tfidf.inverse_transform(x)

    for feature_iter in doc_cor[0]:
        temp_=feature_name.index(feature_iter)
        weight=f_list[0][temp_]
        if feature_iter in doc:
            in_feature_list_index=doc.index(feature_iter)
            feature_list[in_feature_list_index]=weight
    #print(feature_list)
    createHTML(test_report, feature_list, "rcatt.html")
    