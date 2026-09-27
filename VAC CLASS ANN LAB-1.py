import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#load built in dataset
data=load_breast_cancer()

x=data.data
y=data.target

print("Dataset shape", x.shape)
print("Number of classes", len(np.unique(y)))

#Train-Train-Split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)    #fit calculate the statisctics (mean/SD)
X_test = scaler.transform(X_test)          #transform performs scaling

#Create a MLP model
mlp=MLPClassifier(
                    hidden_layer_sizes=(128,64)
                    , activation='relu',solver ='adam', 
                    learning_rate_init=0.001,max_iter=500)

#Train Model
mlp.fit(X_train,y_train)
#prediction
y_pred=mlp.predict(X_test)
print(y_pred)

#Evaluation
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)

print("\nperformance Metrics")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)