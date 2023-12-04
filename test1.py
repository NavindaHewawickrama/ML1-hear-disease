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
