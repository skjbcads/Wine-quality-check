import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#step2
data=pd.read_csv(r'D:\Data Science\winequality-red.csv')
##################################
data.head()
data.tail()
data.info()      
pd.set_option('display.max_columns',50)
data.describe()
data.columns
columns=['quality']
for column in columns:
    plt.figure(figsize=(10,8))
    sns.boxplot(x=data["quality"],y=data[column])
    print('\n')

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True)


x=data.drop(['quality'],axis=1)
y=data['quality']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.20,
                                               random_state=0)

from  sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn import metrics
metrics.confusion_matrix(y_test,y_pred)
metrics.accuracy_score(y_test,y_pred)

print(metrics.classification_report(y_test,y_pred))
data.columns
columns=['quality']
for column in columns:
    plt.figure(figsize=(10,8))
    sns.boxplot(x=data["quality"],y=data[column])
    print('\n')

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True)


x=data.drop(['quality'],axis=1)
y=data['quality']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.20,
                                               random_state=0)

from  sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn import metrics
metrics.confusion_matrix(y_test,y_pred)
metrics.accuracy_score(y_test,y_pred)

print(metrics.classification_report(y_test,y_pred))
data.columns
columns=['fixed acidity','volatile acidity','citric acid','residual sugar',
         'chlorides','free sulfur dioxide','total sulfur dioxide','density',
         'pH','sulphates','alcohol','quality']
for column in columns:
    plt.figure(figsize=(10,8))
    sns.boxplot(x=data["quality"],y=data[column])
    print('\n')

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True)


x=data.drop(['quality'],axis=1)
y=data['quality']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                               test_size=0.20,
                                               random_state=0)

from  sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)

from sklearn import metrics
metrics.confusion_matrix(y_test,y_pred)
metrics.accuracy_score(y_test,y_pred)

print(metrics.classification_report(y_test,y_pred))