from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

data = load_breast_cancer
print(data.feature_names) #Feature of the tumor
print(data.target_names)  #Targets of the tumor

#Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(np.array(data.data), np.array(data.target), test_size=0.2)
#Take all the features and put them into a numpy array, take all the targets and put them into a numpy array
#Then split 20% of that into test data, and 80% of the data into training

#Define a classifier
clf = KNeighborsClassifier(n_neighbors=3) #3 nearest neighbors
clf.fit(x_train, y_train)

print(clf.score(x_test, y_test)) #Print the accuracy of the model

