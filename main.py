import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score


with open('heart_failure_clinical_records_dataset.csv', 'r') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  csv_reader = list(csv_reader)
  csv_reader = np.array(csv_reader[1:300]).astype("float")

X=csv_reader[1:300, 0:12]

Y=csv_reader[1:300, 12]

X_train, X_test, y_train, y_test = train_test_split(X,Y , test_size=0.5, random_state = 42, stratify= Y)


gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Algorithm: Gaussaian Naive Bayes\nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print("Acuracia: %f" % (accuracy_score(y_test,y_pred))) 
print("F1-score: %f" % (f1_score(y_test, y_pred, average = 'macro')))
print("Precisao: %f\n" % (precision_score(y_test, y_pred, average = 'macro')))


cnb = ComplementNB()
y_pred = cnb.fit(X_train, y_train).predict(X_test)
print("Algorithm: Complement Naive Bayes\nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print("Acuracia: %f " % (accuracy_score(y_test,y_pred)))
print("F1-score: %f" % (f1_score(y_test, y_pred, average = 'macro')))
print("Precisao: %f\n" % (precision_score(y_test, y_pred, average = 'macro')))


dtc = DecisionTreeClassifier()
y_pred = dtc.fit(X_train,y_train).predict(X_test)
print("Algorithm: Decision Tree \nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print("Acuracia: %f " % (accuracy_score(y_test,y_pred)))
print("F1-score: %f" % (f1_score(y_test, y_pred, average = 'macro')))
print("Precisao: %f\n" % (precision_score(y_test, y_pred, average = 'macro')))
#y_proba = dtc.predict_proba(X_test)
#print(y_proba)


knn = KNeighborsClassifier(6)
y_pred = knn.fit(X_train,y_train).predict(X_test)
print("Algorithm: 6-Nearest neighbors \nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print("Acuracia: %f " % (accuracy_score(y_test,y_pred)))
print("F1-score: %f" % (f1_score(y_test, y_pred, average = 'macro')))
print("Precisao: %f\n" % (precision_score(y_test, y_pred, average = 'macro')))

svm = svm.SVC()
y_pred = svm.fit(X_train,y_train).predict(X_test)
print("Algorithm: Support Vector Machine (SVC without scaler)\nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print("Acuracia: %f " % (accuracy_score(y_test,y_pred)))
print("F1-score: %f" % (f1_score(y_test, y_pred, average = 'macro')))
print("Precisao: %f\n" % (precision_score(y_test, y_pred, average = 'macro', zero_division = 1)))

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
y_pred = clf.fit(X_train, y_train).predict(X_test)
print("Algorithm: Support Vector Classification \nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum())) 
print("Acuracia: %f " % (accuracy_score(y_test,y_pred)))
print("F1-score: %f" % (f1_score(y_test, y_pred, average = 'macro')))
print("Precisao: %f\n" % (precision_score(y_test, y_pred, average = 'macro')))

            
#pegar média e desvio padrão de todos algoritmos
#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
#plotar diferentes gráficos de cada algoritmos
#acuracia, precisão, f1 score, 