import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('kaggle_diabetes.csv')

df = df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

df2 = df.copy(deep=True)
df2[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df2[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)


# Replacing NaN value by mean, median depending upon distribution
df2['Glucose'].fillna(df2['Glucose'].mean(), inplace=True)
df2['BloodPressure'].fillna(df2['BloodPressure'].mean(), inplace=True)
df2['SkinThickness'].fillna(df2['SkinThickness'].median(), inplace=True)
df2['Insulin'].fillna(df2['Insulin'].median(), inplace=True)
df2['BMI'].fillna(df2['BMI'].median(), inplace=True)

# Model Building
from sklearn.model_selection import train_test_split
X = df2.drop(columns='Outcome')
y = df2['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))