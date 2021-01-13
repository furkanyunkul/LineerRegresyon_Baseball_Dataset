# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 13:27:02 2021

@author: Kullanıcı
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score



df = pd.read_csv('baseball.csv')

#Kolon isimlendirme
print(df.columns)
kolon=df.columns
#Shape
print(df.shape)
#Aciklama
print(df.describe())
describe=df.describe()

#2002 yili oncesi verilerimiz train kumesi

df['RD'] = df['RS'] - df['RA']


print(df.isna().sum()/len(df))
"""
df.drop(['RankSeason'],axis=1,inplace=True)
df.drop(['RankPlayoffs'],axis=1,inplace=True)
"""

"""
df.drop(['OOBP'],axis=1,inplace=True)
df.drop(['OSLG'],axis=1,inplace=True)
"""
dfTest=df[df.Year>=2002]
df = df[df.Year < 2002]
describe=df.describe()

correlation=df.corr()
plt.figure(figsize=(16,16))
sns.heatmap(correlation,annot=True)

flatui = ["#6cdae7", "#fd3a4a", "#ffaa1d", "#ff23e5", "#34495e", "#2ecc71"]
sns.set_palette(flatui)
sns.palplot(sns.color_palette())

sns.lmplot(x = "W", y = "RS", fit_reg = False, hue = "Playoffs", data=df,height=7, aspect=1.25)
plt.xlabel("W", fontsize = 20)
plt.ylabel("RS", fontsize = 20)
plt.axvline(99, 0, 1, color = "Black", ls = '--')
plt.show()


#Run Diffrence ile Win 
#plt.axhline(99, 0, 1, color = "k", ls = '--')
x = np.array(df.RD)
y = np.array(df.W)
slope, intercept = np.polyfit(x, y, 1)
abline_values = [slope * i + intercept for i in x]
plt.figure(figsize=(10,8))
plt.scatter(x, y)
plt.plot(x, abline_values, 'r')
plt.title("Slope = %s" % (slope), fontsize = 12)
plt.xlabel("RD", fontsize =20)
plt.ylabel("W", fontsize = 20)

plt.show()

print(np.corrcoef(x,y))

#Secondly: We can use the Seaborn pairplot:

corrcheck = df[['RD', 'W', 'Playoffs']].copy()
g = sns.pairplot(corrcheck, hue = 'Playoffs',vars=["RD", "W"])
g.fig.set_size_inches(14,10)

podesta = df[['OBP','SLG','BA','RS']]
print(podesta.corr())

plt.figure(figsize=(16,16))
sns.heatmap(podesta.corr(),annot=True)


#OOBP OSLG RA ile bağlantisi

df2=df.dropna()
podesta2 = df2[['OOBP','OSLG','RA']]
print(podesta2.corr())
plt.figure(figsize=(16,16))
sns.heatmap(podesta2.corr(),annot=True)


x = np.array(df2.OOBP)
y = np.array(df2.RA)
slope, intercept = np.polyfit(x, y, 1)
abline_values = [slope * i + intercept for i in x]
plt.figure(figsize=(10,8))
plt.scatter(x, y)
plt.plot(x, abline_values, 'r')
plt.title("Eğim = %s" % (slope), fontsize = 12)
plt.xlabel("OOBP", fontsize =20)
plt.ylabel("RA", fontsize = 20)
plt.show()


x = np.array(df2.OSLG)
y = np.array(df2.RA)
slope, intercept = np.polyfit(x, y, 1)
abline_values = [slope * i + intercept for i in x]
plt.figure(figsize=(10,8))
plt.scatter(x, y)
plt.plot(x, abline_values, 'r')
plt.title("Eğim = %s" % (slope), fontsize = 12)
plt.xlabel("OSLG", fontsize =20)
plt.ylabel("RA", fontsize = 20)
plt.show()

#OBP On base percentege ile RS Run Scored Farki


x = np.array(df.OBP)
y = np.array(df.RS)
slope, intercept = np.polyfit(x, y, 1)
abline_values = [slope * i + intercept for i in x]
plt.figure(figsize=(10,8))
plt.scatter(x, y)
plt.plot(x, abline_values, 'r')
plt.title("Eğim = %s" % (slope), fontsize = 12)
plt.xlabel("OBP", fontsize =20)
plt.ylabel("RS", fontsize = 20)
plt.show()


#SLG Slugging percentege ile RS Run Scored Farki


x = np.array(df.SLG)
y = np.array(df.RS)
slope, intercept = np.polyfit(x, y, 1)
abline_values = [slope * i + intercept for i in x]
plt.figure(figsize=(10,8))
plt.scatter(x, y)
plt.plot(x, abline_values, 'r')
plt.title("Eğim = %s" % (slope), fontsize = 12)
plt.xlabel("SLG", fontsize =20)
plt.ylabel("RS", fontsize = 20)
plt.show()


#BA batting averagile RS Run Scored Farki

x = np.array(df.BA)
y = np.array(df.RS)
slope, intercept = np.polyfit(x, y, 1)
abline_values = [slope * i + intercept for i in x]
plt.figure(figsize=(10,8))
plt.scatter(x, y)
plt.plot(x, abline_values, 'r')
plt.title("Slope = %s" % (slope), fontsize = 12)
plt.xlabel("BA", fontsize =20)
plt.ylabel("RS", fontsize = 20)

plt.show()

from statsmodels.api import sm
#Model kurma
x=df[['OBP','SLG','BA']].values
y=df[['RS']].values

RS_model=linear_model.LinearRegression()
RS_model.fit(x,y)   



print(RS_model.intercept_)
print(RS_model.coef_)

#RS = -788.46+2917.42×(OBP)+1637.93×(SLG)-368.97×(BA)

#Model kurma
x = df[['OBP','SLG']].values
y = df[['RS']].values

RS_model = linear_model.LinearRegression()

RS_model.fit(x,y)

print(RS_model.intercept_)
print(RS_model.coef_)


#RS = -804.63+2737.77×(OBP)+1584.91×(SLG)

#Model kurma 
x=df2[['OOBP','OSLG']].values
y=df2[['RA']].values

RA_model=linear_model.LinearRegression()
RA_model.fit(x,y)   


print(RA_model.intercept_)
print(RA_model.coef_)


#RD ile W
x = df[['RD']].values
y = df[['W']].values
# Calling our model object.
W_model = linear_model.LinearRegression()
# Fitting the model.
W_model.fit(x,y)
# Printing model intercept and coefficients.
print(W_model.intercept_)
print(W_model.coef_)

#W = 80.88 + 0.11 ×(RD)


#Tahmin
#RS tahmin
print(int(RS_model.predict([[0.339,0.430]])))
predictRS=RS_model.predict([[0.339,0.430]])
#RA tahmin
print(int(RA_model.predict([[0.307,0.373]])))
predictRA=RA_model.predict([[0.307,0.373]])
#W tahmin
predictRD=int(predictRS)-int(predictRA)

print(int(W_model.predict([[int(predictRD)]])))


#-------------------------------------------------------------------------------
#2002 yılı sonraki için
dfTest=dfTest.sort_values(by=['Year'],ascending='True')

testOBP=dfTest['OBP'].values
testSLG=dfTest['SLG'].values
print(int(RS_model.predict([[testOBP[0],testSLG[0]]])))

#Genel tahmin
pRS=[]
for i in range(0,len(testOBP)):
    sonuc=int(RS_model.predict([[testOBP[i],testSLG[i]]]))
    pRS.append(sonuc)


testOOBP=dfTest['OOBP'].values
testOOSLG=dfTest['OSLG'].values

pRA=[]
for i in range(0,len(testOOBP)):
    sonuc=int(RA_model.predict([[testOOBP[i],testOOSLG[i]]]))
    pRA.append(sonuc)

#test win
#RD=RS-RA oldugu icin
testRD=[]
for i in range(0,len(pRS)):
    sonuc=pRS[i]-pRA[i]
    testRD.append(sonuc)
    
predictWATL=[]

for i in range(0,len(testRD)):
    sonuc=int(W_model.predict([[int(testRD[i])]]))
    predictWATL.append(sonuc)
    
for i in range(0,len(predictWATL)):
    print(predictWATL[i])
    

    





