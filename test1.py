import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


heart_data = pd.read_csv('C:/Users/hewaw/OneDrive/Desktop/ML research/python_test/heart_disease_data.csv')
#heart_data.info()


#checking for missing values
#heart_data.isnull().sum()

#statistical measures of the data
#heart_data.describe()


#checking the distribution of target variable
#print(heart_data['target'].value_counts())

#splitting the features and target


#splitting the features and target

x = heart_data.drop(columns='target',axis=1)
y = heart_data['target']
#print(x,y)


#splitting the data into taining data and test data.

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size = 0.2, stratify=y, random_state = 2)
#print(x.shape,x_train.shape,x_test.shape)

#Model trianing
#logistic regression

model = LogisticRegression()

#training the logitic regression model with training data

model.fit()
