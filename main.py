import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

with open('heart_failure_clinical_records_dataset.csv', 'r') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  csv_reader = list(csv_reader)
  csv_reader = np.array(csv_reader[1:300]).astype("float")

X=csv_reader[1:300, 0:12]

Y=csv_reader[1:300, 12]

X_train, X_test, y_train, y_test = train_test_split(X,Y , test_size=0.2,random_state=0)


gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Algorithm: Gaussaian Naive Bayes\nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

cnb = ComplementNB()
y_pred = cnb.fit(X_train, y_train).predict(X_test)
print("Algorithm: Complement Naive Bayes\nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum())) 

dtc = DecisionTreeClassifier()
y_pred = dtc.fit(X_train,y_train).predict(X_test)
print("Algorithm: Decision Tree \nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
knn = KNeighborsClassifier(5)
y_pred = knn.fit(X_train,y_train).predict(X_test)
print("Algorithm: 9-Nearest neighbors \nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

#pegar média e desvio padrão de todos algoritmos
#plotar diferentes gráficos de cada algoritmo