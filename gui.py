from tkinter import *
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from PIL import ImageTk,Image
from tkinter.filedialog import askopenfilename
import cv2 as cv
import numpy as np
#from check import movement
import pickle,re
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
a=Tk()
a.title("STOCK MARKET PREDICTION")
a.geometry("1000x950")
def Home():
    global f
    f.pack_forget()
    f=Frame(a,bg="skyblue")
    f.pack(side="top",fill="both",expand=True)
    home_label=Label(f,text="HOME SCREEN",font="Helvetica 25 bold",bg="skyblue",bd=5)
    home_label.place(x=250,y=250)
def Nlp():
    global f
    f.pack_forget()
    f=Frame(a,bg="skyblue")
    f.pack(side="top",fill="both",expand=True)
    home_label=Label(f,text="NLP",font="Helvetica 25 bold",bg="skyblue",bd=5)
    home_label.place(x=150,y=100)
    home_label1=Label(f,text="Choose The Date",font="Helvetica 15 bold",bg="skyblue",bd=5)
    home_label1.place(x=250,y=200)
    var11=StringVar()
    e11=Entry(f,textvariable=var11,width=40)
    e11.place(x=450,y=200,height=30)
    predict_button1=Button(f,text="Predict",bg="pink",width=40,command=lambda:movement(var11))
    predict_button1.place(x=350,y=300,height=35)
    predict_button2=Button(f,text="Graph",bg="pink",width=30,command=lambda:show(var11))
    predict_button2.place(x=550,y=300,height=35)
    global home_label2,home_label33,home_label133
    home_label2=Label(f,text="The prediction is:",font="Helvetica 15 bold",bg="skyblue",bd=5)
    home_label2.place(x=320,y=380)
    home_label133=Label(f,text="The actual result is:",font="Helvetica 15 bold",bg="skyblue",bd=5)
    home_label133.place(x=320,y=450)
    home_label33=Label(f,text="The accuracy_score is:",font="Helvetica 15 bold",bg="skyblue",bd=5)
    home_label33.place(x=320,y=500)
def Rfr():
    global f
    f.pack_forget()
    f=Frame(a,bg="skyblue")
    f.pack(side="top",fill="both",expand=True)
    home_label=Label(f,text="RFR",font="Helvetica 25 bold",bg="skyblue",bd=5)
    home_label.place(x=150,y=100)
    home_label6=Label(f,text="Choose The Company",font="Helvetica 15 bold",bg="skyblue",bd=5)
    home_label6.place(x=250,y=200)
    home_label1=Label(f,text="Choose The Date",font="Helvetica 15 bold",bg="skyblue",bd=5)
    home_label1.place(x=250,y=300)
    var12=StringVar()
    e11=Entry(f,textvariable=var12,width=40)
    e11.place(x=450,y=300,height=30)
    p=["AAPL","AMS","AMSN","apple","asd","FB","ge","MSFT","NKE","SNE","TWTR","TYO","WWE","yahoo"]
    global var99
    var99=StringVar()
    var99.set("select")
    option=OptionMenu(f,var99,*p)
    option.place(x=550,y=200)
    predict_button1=Button(f,text="Predict",bg="pink",width=40,command=lambda:priceregression(var12,var99))
    predict_button1.place(x=350,y=370,height=35)
    predict_button2=Button(f,text="Graph",bg="pink",width=30,command=lambda:show2(var12,var99))
    predict_button2.place(x=550,y=370,height=35)
    global home_label3,home_label43,home_label433
    home_label3=Label(f,text="The prediction price is",font="Helvetica 15 bold",bg="skyblue",bd=5)
    home_label3.place(x=320,y=450)
    home_label433=Label(f,text="The actual price is:",font="Helvetica 15 bold",bg="skyblue",bd=5)
    home_label433.place(x=320,y=500)
    home_label43=Label(f,text="The mean_squared_error is:",font="Helvetica 15 bold",bg="skyblue",bd=5)
    home_label43.place(x=320,y=550)
    
   
def Arima():
    global f
    f.pack_forget() 
    f=Frame(a,bg="skyblue")
    f.pack(side="top",fill="both",expand=True)

    home_label=Label(f,text="ARIMA",font="Helvetica 25 bold",bg="skyblue",bd=5)
    home_label.place(x=150,y=100)
    home_label1=Label(f,text="Choose the company",font="Helvetica 15 bold",bg="skyblue",bd=5)
    home_label1.place(x=250,y=200)
    home_label1=Label(f,text="Choose The Date",font="Helvetica 15 bold",bg="skyblue",bd=5)
    home_label1.place(x=250,y=300)
    var13=StringVar()
    e11=Entry(f,textvariable=var13,width=40)
    e11.place(x=450,y=300,height=30)
    p=["AAPL1","AMS1","AMSN1","APPLE1","asd11","FB1","GE1","MSFT1","NKE1","SNE1","TWTR1","TYO1","WWE11","YAHOO1"]
    global var199
    var199=StringVar()
    var199.set("select")
    option=OptionMenu(f,var199,*p)
    option.place(x=550,y=200)
    predict_button1=Button(f,text="predict",bg="pink",width=40,command=lambda:arima(var13,var199))
    predict_button1.place(x=350,y=370,height=35)
    global home_label4
    home_label4=Label(f,text="The prediction price is:",font="Helvetica 15 bold",bg="skyblue",bd=5)
    home_label4.place(x=320,y=450)
def movement(var11):
 a=[]
 date=var11.get()

 df = pd.read_csv("news2.csv")
 v=df[df['Date'] == date].values
 v1=df[df['Date'] == date]
 print(v1)
 slicedData=v[0]

 g=slicedData[1]


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
 a.append(slicedData[25])
 a.append(slicedData[26])
 a=np.array(a)
 a=a.reshape(1,25)
 
 df = pd.DataFrame(a, columns = ['Top1','Top2','Top3','Top4','Top5','Top6','Top7','Top8','Top9','Top10','Top11','Top12','Top13','Top14','Top15','Top16','Top17','Top18','Top19','Top20','Top21','Top22','Top23','Top24','Top25'])

 test= df.iloc[:,0:25]
 print(df)

 print(test)
 basicvectorizer = CountVectorizer(ngram_range=(1,1))
 testheadlines = []
 for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,0:25]))
 with open('count.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
 basictest = tokenizer.transform(testheadlines)
 f=open("rfclass.pickle",'rb')
 classifier=pickle.load(f)
 val=classifier.predict(basictest)

 if val[0]==1:
    print("Result is rise")
    v="Result is rise"
    home_label2.config(text=v)
    
 if val[0]==0:
    print("Result is fall")  
    v="Result is fall"
    home_label2.config(text=v)
 if g==1:
    g='The actual result is rise'
 else:
    g='The actual result is fall'
 
 home_label133.config(text=g)
 
 
   
 
def show(var11):
 date=var11.get()
 df1 = pd.read_csv("news2.csv")
 v=df1[df1['Date'] == date].values
 slicedData=v[0]

 global g1
 test1 = df1[df1['Date'] >date ]
 testheadlines1=[]
 for row in range(0,len(test1.index)):
    testheadlines1.append(' '.join(str(x) for x in test1.iloc[row,0:25]))
 with open('count.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
 basictest = tokenizer.transform(testheadlines1)
 f=open("rfclass.pickle",'rb')
 classifier=pickle.load(f)
 predictions=classifier.predict(basictest)
 from sklearn.metrics import classification_report
 from sklearn.metrics import f1_score
 from sklearn.metrics import accuracy_score 
 from sklearn.metrics import confusion_matrix
 matrix=confusion_matrix(test1['Label'],predictions)
 accuracy=accuracy_score(test1['Label'],predictions)
 ac=str(accuracy)
 a='accuracy is :'+ac
 home_label33.config(text=a)
 
 print (confusion_matrix(test1['Label'],predictions))
 print (accuracy_score(test1['Label'],predictions))
 import seaborn as sns
 import matplotlib.pyplot as plt
 ax = sns.heatmap(matrix, annot=True, cmap='Blues')

 ax.set_title('Seaborn Confusion Matrix with labels\n\n');
 ax.set_xlabel('\nPredicted Values')
 ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
 ax.xaxis.set_ticklabels(['False','True'])
 ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
 plt.show()
def priceregression(var12,var99):
 a=[]
 var=var12.get()
 v=var99.get()
 v1="Dataset"+"/"+v+"."+"csv"
 print(v1)
 df = pd.read_csv(v1)
 df1 = pd.read_csv(v1,encoding = "ISO-8859-1")
 v=df[df['Date'] == var].values
 v1=v[:1]
 v2=v1[0]
 v3=v2[1]
 v4=v2[2]
 v5=v2[3]
 v7=v2[5]
 v8=v2[6]
 v11=v2[4]
 print(v11,"kkk")
 a.append(v3)
 a.append(v4)
 a.append(v5)
 a.append(v7)
 a.append(v8)
 a=np.array(a)
 a=a.reshape(1,5)
#print(predict.shape)
 with open('scale1.pickle', 'rb') as handle:
    scale = pickle.load(handle)
 basictest = scale.transform(a)
 f=open("rfregressn1.pickle",'rb')
 classifier=pickle.load(f)
 val=classifier.predict(basictest)
 print(val[0])
 v=val[0]
 v='the predicted price is:'+str(v)
 v11='the actual price is:'+str(v11)
 home_label3.config(text=v)
 home_label433.config(text=v11)
def show2(var12,var99):
 a=[]
 var=var12.get()
 v=var99.get()
 v1="Dataset"+"/"+v+"."+"csv"
 print(v1)
 df1 = pd.read_csv(v1,encoding = "ISO-8859-1")
 
 var1=[]
 var=var.split('/')

 var=''.join(var)
 
 test1= df1[df1['Date']>var]


 X=test1.drop('Date',axis=1)
 test2=X['Close'].tolist()
 
 x=X.drop('Close',axis=1)
 predictions1= []
 testheadlines=[]
 a5=[]
 for row in range(0,len(x.index)):
    testheadlines.append(x.iloc[row,0:5])
   

 with open('scale1.pickle', 'rb') as handle:
     scale1 = pickle.load(handle)
 basictest11 = scale1.transform(testheadlines)
 a5=[]
 f=open("rfregressn1.pickle",'rb')
 classifier1=pickle.load(f)
 predictions1=classifier1.predict(basictest11)
 # print(len(testheadlines))
  
 from sklearn.metrics import classification_report
 from sklearn.metrics import f1_score
 from sklearn.metrics import accuracy_score 
 from sklearn.metrics import confusion_matrix
 from sklearn import metrics

 # matrix=confusion_matrix(test2,predictions1)

 ac=round(metrics.mean_absolute_error(test2, predictions1),4)
 
 print(test2,"test")
 print(predictions1,"pres")
 print(type(ac))
 ac=str(ac)
 a='The mean_squared_error is :'+ac
 print(a)
 home_label43.config(text=a)
 import matplotlib.pyplot as plt
 plt.plot(test2,color='red',linewidth=3,label='actual values')
 plt.plot(predictions1,color='blue',label='predictions')
 plt.xlabel('counts')
# naming the y axis
 plt.ylabel('values')
 plt.legend(loc='best')
 plt.show()

def arima(var13,var199):
    var=var13.get()
    v=var199.get()
    v1=v+"."+"csv"
    df = pd.read_csv(v1)
  
    y=df[['Close']].copy()
    count = df.shape[0]
    n1=var13.get()
    n2=df[df['Date'] ==n1].index.values
    n3=n2[0]
    n4=n3+2
    print(n4)
    # n=int(len(df)*.8)
   
    train=y[:n4]
    test=y[n4:]
    print(count)
    v=df['Close'][n4]
    print(v)
    c=count
    h1=c-n4
    print(h1)
    from statsmodels.tsa.arima_model import ARIMA
    import matplotlib.pyplot as plt
    model=ARIMA(train,order=(6,1,3))
    result=model.fit(disp=0)
    fc, se, conf = result.forecast(h1, alpha=0.05)
   
   
    # plt.show()
   
    print(result.summary())
    step=1
    fc=result.forecast(step)
    print(fc[0])
    v=fc[0]
    v1=v[0]
    home_label4.config(text=v1)           
f=Frame(a,bg="cyan")
f.pack(side="top",fill="both",expand=True)
home_label=Label(f,text="HOME SCREEN",font="Helvetica 35 bold",bg="cyan",bd=5)
home_label.place(x=250,y=250)
m=Menu(a)
m.add_command(label="Home",command=Home)
checkmenu=Menu(m)
m.add_command(label="NLP",command=Nlp)
m.add_command(label="RFR",command=Rfr)
m.add_command(label="ARIMA",command=Arima)
a.config(menu=m)
a.mainloop()