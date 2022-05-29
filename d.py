import pandas as pd
import numpy as np
d1=pd.read_csv('Dataset/AMZN.csv')

d2=d1.loc[::-1].reset_index(drop = True)
print(d2)
d2.to_csv('AMZN1.csv', header=False, index=False)
d3=pd.read_csv('Dataset/apple.csv')

d4=d3.loc[::-1].reset_index(drop = True)
print(d2)
d4.to_csv('apple.csv', header=False, index=False)
d5=pd.read_csv('Dataset/asd.csv')

d6=d5.loc[::-1].reset_index(drop = True)
print(d2)
d6.to_csv('apple1.csv', header=False, index=False)
d7=pd.read_csv('Dataset/asd.csv')

d8=d7.loc[::-1].reset_index(drop = True)
print(d2)
d8.to_csv('asd11.csv', header=False, index=False)
d9=pd.read_csv('Dataset/FB.csv')

d10=d9.loc[::-1].reset_index(drop = True)
print(d2)
d10.to_csv('FB1.csv', header=False, index=False)
d11=pd.read_csv('Dataset/ge.csv')

d12=d11.loc[::-1].reset_index(drop = True)
print(d2)
d12.to_csv('ge1.csv', header=False, index=False)
d13=pd.read_csv('Dataset/MSFT.csv')

d14=d13.loc[::-1].reset_index(drop = True)
print(d2)
d14.to_csv('MSFT1.csv', header=False, index=False)
d15=pd.read_csv('Dataset/NKE.csv')

d16=d15.loc[::-1].reset_index(drop = True)
print(d2)
d16.to_csv('NKE1.csv', header=False, index=False)
d17=pd.read_csv('Dataset/SNE.csv')

d18=d17.loc[::-1].reset_index(drop = True)
print(d2)
d18.to_csv('AMZN1.csv', header=False, index=False)
d19=pd.read_csv('Dataset/TWTR.csv')

d20=d19.loc[::-1].reset_index(drop = True)
print(d2)
d20.to_csv('TWTR1.csv', header=False, index=False)
d21=pd.read_csv('Dataset/TYO.csv')

d23=d21.loc[::-1].reset_index(drop = True)
print(d2)
d23.to_csv('TYO1.csv', header=False, index=False)
d24=pd.read_csv('Dataset/WWE.csv')

d25=d24.loc[::-1].reset_index(drop = True)
print(d2)
d25.to_csv('WWE11.csv', header=False, index=False)
d26=pd.read_csv('Dataset/yahoo.csv')

d27=d26.loc[::-1].reset_index(drop = True)
print(d2)
d27.to_csv('yahoo1.csv', header=False, index=False)
