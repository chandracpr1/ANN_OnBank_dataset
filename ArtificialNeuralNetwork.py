import numpy as np
import tensorflow as tf
import pandas as pd
tf.__version__

dataset=pd.read_csv("Churn_Modelling.csv")
X=dataset.iloc[:,3:-1].values
y=dataset.iloc[:,-1].values
print(X)
print(y)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X[:,2]=le.fit_transform(X[:,2])
print(X)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# compiling ANN

ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# training the ANN on the training set

ann.fit(X_train, y_train,batch_size=32,epochs=100)


# Predicting the Test set results

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix,accuracy_score
cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred)
print(cm)
accuracy_score(y_test, y_pred)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(2,2))
ax=sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('truth')



