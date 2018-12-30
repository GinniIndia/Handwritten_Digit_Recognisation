#Program to Recognise Hand-Written Digit of MNIST Data-set

#Build-in Libraries
import numpy as np
import matplotlib.pyplot as pt
from sklearn.tree import DecisionTreeClassifier
import pandas as pn

#reading data from file
data=pn.read_csv('train.csv').as_matrix()
clf=DecisionTreeClassifier()

#training data set
xtrain=data[0:9000,1:]
train_label=data[0:9000,0]
clf.fit(xtrain,train_label)

#testing data set
xtest=data[4000:,1:]
actual_label=data[4000:,0]
d=xtest[5000]
d.shape=(28,28)
pt.imshow(d,cmap='gray')
print('Predict',clf.predict([xtest[5000]]))
pt.show()

#counting the Percentage Accuracy
count=0
for i in range(2000):
	value=clf.predict([xtest[i]])
	if(value==actual_label[i]):
		count=count+1
print('Percentage Accuracy',count/2000*100);
