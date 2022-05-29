##############stock market randomclassifier##############
########################################################
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
#load dataset
data = pd.read_csv('news2.csv', encoding = "ISO-8859-1")
data.head(1)
#########################################
#splitting dataset for testing and traing
train = data[data['Date'] < '20150101']
test = data[data['Date'] > '20141231']
slicedData= train.iloc[:,2:27]
slicedData.replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
slicedData.columns= new_Index
print(slicedData.head(5))
for index in new_Index:
    slicedData[index]=slicedData[index].str.lower()
print(slicedData.head(1))
headlines = []
for row in range(0,len(slicedData.index)):
    headlines.append(' '.join(str(x) for x in slicedData.iloc[row,0:25]))
basicvectorizer = CountVectorizer(ngram_range=(1,1))
basictrain = basicvectorizer.fit_transform(headlines)

print(basictrain.shape)
#############################
#model creation
basicmodel = RandomForestClassifier(n_estimators=100,random_state=42, criterion='entropy',max_features='auto')
basicmodel = basicmodel.fit(basictrain, train["Label"])
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
basictest = basicvectorizer.transform(testheadlines)
#############################
#model testing
predictions = basicmodel.predict(basictest)
##############################
#accuracy 
print(pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"]))
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix

print (classification_report(test["Label"], predictions))
print (accuracy_score(test["Label"], predictions))
##############################################
#####model saving
f=open("rfclass.pickle","wb")
pickle.dump(basicmodel,f)
f.close()
#############################################
######saving countvector
with open('count.pickle', 'wb') as handle:
    pickle.dump(basicvectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
