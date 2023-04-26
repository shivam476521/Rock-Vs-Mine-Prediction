# Rock-Vs-Mine-Prediction
#Importing the Dependencies


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection and Data Processing




sonar_data=pd.read_csv('/content/sonar data.csv',header=None)

sonar_data.head()

# Number of rows and columns
sonar_data.shape

sonar_data.describe()  #describe statistical mesures of data

sonar_data[60].value_counts()

# Separating data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

print(X)
print(Y)


# Training and Test Data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.07, stratify=Y, random_state=1 )

print(X.shape,X_train.shape,X_test.shape)

print(Y.shape, Y_train.shape, Y_test.shape)

# Model Training->Logistic Regression


model=LogisticRegression()

model.fit(X_train,Y_train)

# Model Evaluation

# accuracy of model on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# accuracy of model on test data
X_test_prediction=model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print("Accuracy on test data : ", test_data_accuracy)
print("Accuracy on training data : ", training_data_accuracy)

input_data=(0.02,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.066,0.2273,0.31,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.555,0.6711,0.6415,0.7104,0.808,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.051,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.018,0.0084,0.009,0.0032)
# Changing the Input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)
# we need to reshape the numpy array as we are predicting for an instance
input_data_reshaped=input_data_as_numpy_array.reshape(1, -1)
prediction=model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]=='M'):
  print("The object is a mine.")
else:
  print("The object is a rock.")
