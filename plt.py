
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
def f2_score(p,r):
    if p+r==0:
        return 0
    return (5*p*r)/(4*p+r)

data=[["113 TTPs",0.22,0.20,0.42,0.36,0.49,0.55],
      ["46 TTPs",0.21,0.35,0.50,0.45,0.55,0.62],
      ["23 TTPs",0.23,0.45,0.56,0.50,0.63,0.71]
     ]

df=pd.DataFrame(data,columns=["TTP num","HAN","rcATT","TRAM","TTPdrill","COMA","MITretrieval"])
df.plot(x="TTP num", y=["HAN","rcATT","TRAM","TTPdrill","COMA","MITretrieval"], kind="bar",figsize=(9,8),rot=0)
plt.savefig("Overall Comparison_macro.jpg")

'''
data=[["113 TTPs",0.48,0.07],
      ["46 TTPs",0.52,0.1],
      ["23 TTPs",0.61,0.1]
     ]

df=pd.DataFrame(data,columns=["TTP num","Before","After"])
df.plot(x="TTP num", y=["Before","After"], kind="bar",figsize=(9,8), stacked=True,rot=0)
plt.savefig("The effect of Fusion macro.jpg")
'''
'''
#For every F2 score comparison
np_=np.genfromtxt('TTP_study/dl_multihead_40.csv',delimiter=',')
np_1=np.genfromtxt('newbert_25result2.csv',delimiter=',')
np_2=np.genfromtxt('rcATT_result2_40.csv',delimiter=',')
np_3=np.genfromtxt('HAN/HAN_result2_0.csv',delimiter=',')
f2_score1=[]
f2_score2=[]
f2_score3=[]
f2_score4=[]
index=[]
index1=[]
index2=[]
index3=[]
for id,iter_np in enumerate(np_):
    index.append(id)
    p=iter_np[1]
    r=iter_np[2]
    f2_score1.append(f2_score(p,r))
for id,iter_np in enumerate(np_1):
    index1.append(id)
    p=iter_np[1]
    r=iter_np[2]
    f2_score2.append(f2_score(p,r))
for id,iter_np in enumerate(np_2):
    index2.append(id)
    p=iter_np[1]
    r=iter_np[2]
    f2_score3.append(f2_score(p,r))
for id,iter_np in enumerate(np_3):
    index3.append(id)
    p=iter_np[1]
    r=iter_np[2]
    f2_score4.append(f2_score(p,r))

plt.plot(index,f2_score1)
plt.plot(index1,f2_score2)
plt.plot(index2,f2_score3)
plt.plot(index3,f2_score4)
plt.legend(['Token-based','Sentence-based','rcATT','HAN'])
plt.xlabel("Technique")
plt.ylabel("F2 Score")
plt.savefig("40_Tech_com.jpg")
'''

'''
with open("loss_file_22","rb") as f:
    loss=pickle.load(f)
with open("loss_file_sen550","rb") as f:
    loss2=pickle.load(f)
f.close()
index=[i for i in range(100)]
index1=[i for i in range(101)]
plt.plot(index,loss)
plt.plot(index1,loss2)
plt.legend(['Token-based','Sentence-based'])
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.savefig("Loss_80.jpg")
'''
'''
# The effect of fusion
np_=np.genfromtxt('clean_ver_40.csv',delimiter=',')
np_1=np.genfromtxt('no_clean_version_40.csv',delimiter=',')

f2_score1=[]
f2_score2=[]

index=[]
index1=[]

for id,iter_np in enumerate(np_):
    index.append(id)
    p=iter_np[1]
    r=iter_np[2]
    f2_score1.append(f2_score(p,r))
for id,iter_np in enumerate(np_1):
    index1.append(id)
    p=iter_np[1]
    r=iter_np[2]
    f2_score2.append(f2_score(p,r))


plt.plot(index,f2_score1)
plt.plot(index1,f2_score2)

plt.legend(['With Preprocess','Without Preprocess'])
plt.xlabel("CTI Report")
plt.ylabel("F2 Score")
plt.savefig("effect_of_fusion_40_preprocess.jpg")
'''

'''
with open("loss_file_22","rb") as f:
    loss=pickle.load(f)
with open("loss_file_sen550","rb") as f:
    loss2=pickle.load(f)
f.close()
index=[i for i in range(100)]
index1=[i for i in range(101)]
plt.plot(index,loss)
plt.plot(index1,loss2)
plt.legend(['Token-based','Sentence-based'])
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.savefig("Loss_80.jpg")
'''