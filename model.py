# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

#Load dataset
data = pd.read_csv('heart.csv')
print(data.head())
print(data.columns)
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,fmt='.1f')
plt.show()

#test train split
x = data.iloc[:,:-1]
y = data.iloc[:,13]
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print('X_train',X_train.shape)
print('X_test',X_test.shape) 
print('y_train',y_train.shape)
print('y_test',y_test.shape)
# Normalize
X_train=(X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train)).values
X_test=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test)).values

#GaussianNB for classification
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred=gnb.predict(X_test)
Score = gnb.score(X_train, y_train)
print('Accurancy :',accuracy_score(y_test, y_pred))

print("GaussianNB TRAIN score ",format(gnb.score(X_train, y_train)))

cm=confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(cm,annot=True)
plt.show()


# Saving model to disk
pickle.dump(gnb, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(gnb.predict([[63,1,3,145,233,1,0,150,0,2.3,0,0,1]]))