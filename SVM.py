#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:41:49 2019

@author: yugyeongkim
"""

import operator
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets.samples_generator import make_blobs


data_adult = pd.read_csv("Autism-Adult-Data.CSV")
data_adol = pd.read_csv("Autism-Adolescent-Data.CSV")
data_child = pd.read_csv("Autism-Child-Data.CSV")

#replace ? with mean value 2
#df_adult = data_adult[data_adult.ethnicity != "?"]
df_adult = data_adult.replace("?",2)
#df_adult["ethnicity"].mean()
df_adol = data_adol.replace("?",2)
df_child = data_child.replace("?",2)


#encoding categorical values for adults
no_gender = {"gender": {"f":0,"m":1}}
no_jund = {"jundice":{"yes":0,"no":1}}
no_class = {"Class/ASD" : {"YES":0,"NO":1}}
no_asd = {"austim": {"yes":0,"no":1}}



#counts=df_adult["contry_of_res"].value_counts()

df_adult= df_adult.drop(["used_app_before","age_desc","relation","ethnicity","contry_of_res"], axis=1)
df_adult.replace(no_jund,inplace=True)
df_adult.replace(no_class,inplace=True)
df_adult.replace(no_gender,inplace=True)
df_adult.replace(no_asd,inplace=True)

#encoding values for adol
df_adol= df_adol.drop(["used_app_before","age_desc","relation","contry_of_res","ethnicity"], axis=1)
df_adol.replace(no_jund,inplace=True)
df_adol.replace(no_class,inplace=True)
df_adol.replace(no_gender,inplace=True)
df_adol.replace(no_asd,inplace=True)



#encoding for child

df_child= df_child.drop(["used_app_before","age_desc","relation","contry_of_res","ethnicity"], axis=1)
df_child.replace(no_jund,inplace=True)
df_child.replace(no_class,inplace=True)
df_child.replace(no_gender,inplace=True)
df_child.replace(no_asd,inplace=True)

#spliting data into training and test sets.

X_adult = df_adult.drop('Class/ASD',axis=1)
y_adult = df_adult['Class/ASD']
X_adult_train,X_adult_test,y_adult_train,y_adult_test = train_test_split(X_adult,y_adult,test_size = 0.20)

X_adol = df_adol.drop('Class/ASD',axis=1)
y_adol = df_adol['Class/ASD']
X_adol_train,X_adol_test,y_adol_train,y_adol_test = train_test_split(X_adol,y_adol,test_size = 0.20)

X_child = df_adol.drop('Class/ASD',axis=1)
y_child = df_adol['Class/ASD']
X_child_train,X_child_test,y_child_train,y_child_test = train_test_split(X_child,y_child,test_size = 0.20)
#Running SVM
#svclassifier = SVC(kernel='linear')
svclassifier = SVC(kernel='rbf')
svclassifier.fit(X_adult_train,y_adult_train)
svclassifier.fit(X_adol_train,y_adol_train)
svclassifier.fit(X_child_train,y_child_train)
#making predictions
y_adult_pred = svclassifier.predict(X_adult_test)
y_adol_pred = svclassifier.predict(X_adol_test)
y_child_pred = svclassifier.predict(X_child_test)
#evaluation
print("-----------Evaluation for Adults-------------")
print(confusion_matrix(y_adult_test,y_adult_pred))
print(classification_report(y_adult_test,y_adult_pred))


print("-----------Evaluation for Adolescent----------")
print(confusion_matrix(y_adol_test,y_adol_pred))
print(classification_report(y_adol_test,y_adol_pred))



print("-----------Evaluation for Child----------")
print(confusion_matrix(y_child_test,y_child_pred))
print(classification_report(y_child_test,y_child_pred))


fig,ax = plt.subplots()
ax.bar(df_child["austim"],df_child["Class/ASD"])

