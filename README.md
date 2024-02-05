# MITREtrieval
This project aims to extract TTPs from the input CTI report. 
## Model Downloads

Download the [SRL model, SBERT model, and relevant ontology vector files](https://drive.google.com/drive/folders/10HotHLs_h_Oy4IJbbC_Ln26UG7NJuZrm?usp=drive_link).
## Quick Start
Currently only support **technique threshold 6**.
If you want other threshold, you can retrain model(Jump to [here](#retrain-model))
Below is the command for predicting your own CTI report.
MITREtreival will cost about **3-10 minutes** considering the length of input CTI reports

---
### Step1:
Put the CTI reports into input.txt file

### Step2:
```
/*stage: What you want to do
 *threshold: Techinque threshold
*/
 python main.py --stage MITREtreival --threshold 6 
```

## Step3:
Get corresponding techniques(ID) in terminal


## Retrain model
The whole retraining process is divide into different phases because it might take a long time if combine every components.

---
### Step1: Spliting Dataset
In the part, you can get training and testing set depending on technique threshold(You can designated).
**To split dataset**
```
 python main.py --stage train_test_split --threshold 6 
```
### Step2: Topic Classifier
In this part, irrelevant sentences will be removed by our pretrained model
**To use Topic Classifier on training and testing set**
```
 python main.py --stage preprocess --threshold 6 
```

**To retrain topic classifier:**
```
 python filter_bert.py train //Train
 python filter_bert.py train //Test
```

**To add more data into topic classifier dataset**
Add more info in **filter_bert.ipynb** to build topic classifier dataset

### Step3: Knowledge Expansion
In Knowledge Expansion, there are 2 phases, Extract query nodes and query COMAT.

**To extract nodes:**
```
python main.py --stage feature_extraction --threshold 6 
```

**To query COMAT:**
```
python main.py --stage Inference_train --threshold 6 //Query Training set
python main.py --stage Inference --threshold 6 //Query Testing set
```
### Step4: Model Training
You will get **sentence_BERT_model_threshold.pt** model
**Train whole model(With knowledge fusion):**
```
python main.py --stage model_fusion --threshold 6 
```
**Train model(Without knowledge fusion):**
```
python main.py --stage model_train_sen --threshold 6 
```
