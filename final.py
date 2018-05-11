# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 18:29:53 2018

@author: arka
"""

import numpy as np
import pandas as pd
df = pd.read_csv('eithnicuty_ground__truth.csv')

df['freq_surname'] = df.groupby('Surname')['Surname'].transform('count') 
df['freq_name'] = df.groupby('Name')['Name'].transform('count')


###############################################################################
#data preprocessing
#vectorising

community={}
for i in df['Community']:
     if i in community.keys():
         community[i]+=1
     else:
         community[i]=1
             
name_community={}
for i in range(len(df['Name'])):
    if df['Name'][i] not in name_community.keys():
        name_community[df['Name'][i]]=[df['Community'][i]]
    else:
        name_community[df['Name'][i]]+=[df['Community'][i]]
        
name_community_count={}
for i in name_community.keys():
    name_community_count[i]={}
    for j in community:
        name_community_count[i][j]=name_community[i].count(j)/len(name_community[i])
        
        
surname_community={}
for i in range(len(df['Surname'])):
    if df['Surname'][i] not in surname_community.keys():
        surname_community[df['Surname'][i]]=[df['Community'][i]]
    else:
        surname_community[df['Surname'][i]]+=[df['Community'][i]]
        
surname_community_count={}
for i in surname_community.keys():
    surname_community_count[i]={}
    for j in community:
        surname_community_count[i][j]=surname_community[i].count(j)/len(surname_community[i])
 

       
#putting the conditional probabilities in original dataset        
for i in community:
    for j in range(df['Name'].size):
        df.at[j,'name_'+i]=name_community_count[df.at[j,'Name']][i]
        df.at[j,'surname_'+i]=surname_community_count[df.at[j,'Surname']][i]
        

###############################################################################
#splitting dataset into training and testing sets

X=df.iloc[:,6:]     
X=X.as_matrix()

from sklearn.preprocessing import LabelEncoder
y=df.iloc[:,3]
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


###############################################################################
#model training and fitting


#need to convert feature columns(that is, y here) to int64 otherwise fit function will give an error
X0=np.array(X_train, dtype = 'float32') 
y0=np.array(y_train, dtype = 'int64') 


X1=np.array(X_test, dtype = 'float32') 
y1=np.array(y_test, dtype = 'int64') 



import tensorflow as tf
#feature_columns for DNNClassifier
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=54)]

#DNNClassifier from learn
import tensorflow.contrib.learn as learn
classifier = learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[100, 200, 100], n_classes=27)

#fitting classifier to training data
classifier.fit(X0,y0,steps=5000, batch_size=20)

accuracy_calc= classifier.evaluate(X0, y0)["accuracy"]

print('\n')
print('Accuracy: {0:f}'.format(accuracy_calc))
print('\n')


#####################################################
#same as accuracy of prediction as observed later
accuracy_calc1= classifier.evaluate(X1, y1)["accuracy"]

print('\n')
print('Accuracy: {0:f}'.format(accuracy_calc1))
print('\n')

#################################################


y_predicted = classifier.predict(X_test)
y1_predicted = classifier.predict(X_test)
  




#evaluating using testing data
from sklearn.metrics import classification_report

print(classification_report(y_test,list(y_predicted), digits=2))
print('\n')


#it is ibserved that accuracy of prediction is the same as accuracy obtained on evaluating our model on test set.
from sklearn.metrics import accuracy_score
print('\n')
print('Accuracy of prediction : ',accuracy_score(y_test,list(y1_predicted)))