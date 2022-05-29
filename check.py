import pickle,re
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
def movement(var11):
 a=[]
 print(var11)
 df = pd.read_csv("Dataset/apples.csv")
 v=df[df['Date'] == Data].values
 print(v)
 print(v[2:])
 Data=str(input("enter clumb_thickness :"))
 slicedData= Data.split(',')
 a.append(slicedData[0])
 a.append(slicedData[1])
 a.append(slicedData[2])
 a.append(slicedData[3])
 a.append(slicedData[4])
 a.append(slicedData[5])
 a.append(slicedData[6])
 a.append(slicedData[7])
 a.append(slicedData[8])
 a.append(slicedData[9])
 a.append(slicedData[10])
 a.append(slicedData[11])
 a.append(slicedData[12])
 a.append(slicedData[13])
 a.append(slicedData[14])
 a.append(slicedData[15])
 a.append(slicedData[16])
 a.append(slicedData[17])
 a.append(slicedData[18])
 a.append(slicedData[19])
 a.append(slicedData[20])
 a.append(slicedData[21])
 a.append(slicedData[22])
 a.append(slicedData[23])
 a.append(slicedData[24])
 a=np.array(a)
 a=a.reshape(1,25)
 
 df = pd.DataFrame(a, columns = ['Top1','Top2','Top3','Top4','Top5','Top6','Top7','Top8','Top9','Top10','Top11','Top12','Top13','Top14','Top15','Top16','Top17','Top18','Top19','Top20','Top21','Top22','Top23','Top24','Top25'])

 test= df.iloc[:,0:25]
 print(slicedData)
 basicvectorizer = CountVectorizer(ngram_range=(1,1))
 testheadlines = []
 for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
 with open('count.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
 basictest = tokenizer.transform(testheadlines)
 f=open("rfclass.pickle",'rb')
 classifier=pickle.load(f)
 val=classifier.predict(basictest)
 print(val)
 if val[0]==1:
    print("Result is rise")
 if val[0]==0:
    print("Result is fall")  
if __name__=="__main__":
 movement()  
