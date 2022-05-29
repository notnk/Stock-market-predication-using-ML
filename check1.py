import pickle
import pandas as pd
import numpy as np
def priceregression():
 a=[]
 n=input("date")
 df = pd.read_csv("Dataset/yahoo.csv")
 v=df[df['Date'] == n].values
 v1=v[:1]
 v2=v1[0]
 v3=v2[1]
 v4=v2[2]
 v5=v2[3]
 v7=v2[5]
 v8=v2[6]
#ct=float(input("enter open :"))
 a.append(v3)
#su=float(input("enter high :"))
 a.append(v4)
#shu=float(input("enter low :"))
 a.append(v5)
#ma=float(input("enter volume :"))
 a.append(v7)
#ma1=float(input("enter adj close :"))
 a.append(v8)
 a=np.array(a)
 a=a.reshape(1,5)
#
#print(predict.shape)
 with open('scale1.pickle', 'rb') as handle:
    scale = pickle.load(handle)
 basictest = scale.transform(a)
 f=open("rfregressn1.pickle",'rb')
 classifier=pickle.load(f)
 val=classifier.predict(basictest)
 print(val)
if __name__=="__main__":
 priceregression()  
