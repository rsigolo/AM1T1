import csv
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score,classification_report,roc_curve,auc

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open('/home/augusto/ufscar/ENPE Bloco B/AM/heart_failure_clinical_records_dataset.csv', 'r') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  csv_reader = list(csv_reader)
  csv_reader = np.array(csv_reader[1:300]).astype("float")

X=csv_reader[1:300, 0:12]

Y=csv_reader[1:300, 12]
X_train, X_test, y_train, y_test = train_test_split(X,Y , test_size=0.5, random_state = 42, stratify= Y)

results = pd.DataFrame()
model_precision = []
model_accuracy = []
model_f1_score = []
model_auc = []
roc_curves = []
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Algorithm: Gaussian Naive Bayes(without scaler)\nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print("Acuracia: %f" % (accuracy_score(y_test,y_pred))) 
print("F1-score: %f" % (f1_score(y_test, y_pred, average = 'macro')))
print("Precisao: %f\n" % (precision_score(y_test, y_pred, average = 'macro')))
print(classification_report(y_test,y_pred))

clf = make_pipeline(StandardScaler(),gnb)
params = {
'standardscaler__with_mean':[False,True],
'standardscaler__with_std':[False,True],
'gaussiannb__var_smoothing':[1e-9,1e-8,1e-7,1e-6]
}
grid1 = GridSearchCV(clf,params)
y_pred = grid1.fit(X_train, y_train).predict(X_test)
print("Algorithm: Gaussaian Naive Bayes\nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print("Acuracia: %f" % (accuracy_score(y_test,y_pred)))
model_accuracy.append(accuracy_score(y_test,y_pred))
print("F1-score: %f" % (f1_score(y_test, y_pred, average = 'macro')))
model_f1_score.append(f1_score(y_test, y_pred, average = 'macro'))
print("Precisao: %f" % (precision_score(y_test, y_pred, average = 'macro')))
model_precision.append(precision_score(y_test, y_pred, average = 'macro'))
print("Melhores parametros:",grid1.best_params_,'\n')
print(classification_report(y_test,y_pred))
roc_curves.append(roc_curve(y_test,grid1.predict_proba(X_test)[:,1]))

cnb = ComplementNB()
params = {
  'alpha': [1,2,3]
}
grid = GridSearchCV(cnb,params)
y_pred = grid.fit(X_train, y_train).predict(X_test)
print("Algorithm: Complement Naive Bayes(without scaler)\nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print("Acuracia: %f " % (accuracy_score(y_test,y_pred)))
print("F1-score: %f" % (f1_score(y_test, y_pred, average = 'macro')))
print("Precisao: %f\n" % (precision_score(y_test, y_pred, average = 'macro')))
print(classification_report(y_test,y_pred))
print("Melhores parametros:",grid.best_params_,'\n')


clf = make_pipeline(StandardScaler(with_mean=False),cnb)
params = {
'standardscaler__with_std':[False,True],
'complementnb__alpha': [1,2,3]
}
grid2 = GridSearchCV(clf,params)
y_pred = grid2.fit(X_train, y_train).predict(X_test)
print("Algorithm: Complement Naive Bayes\nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print("Acuracia: %f" % (accuracy_score(y_test,y_pred)))
model_accuracy.append(accuracy_score(y_test,y_pred))
print("F1-score: %f" % (f1_score(y_test, y_pred, average = 'macro')))
model_f1_score.append(f1_score(y_test, y_pred, average = 'macro'))
print("Precisao: %f" % (precision_score(y_test, y_pred, average = 'macro')))
model_precision.append(precision_score(y_test, y_pred, average = 'macro'))
print("Melhores parametros:",grid2.best_params_,'\n')
print(classification_report(y_test,y_pred))
roc_curves.append(roc_curve(y_test,grid2.predict_proba(X_test)[:,1]))


dtc = DecisionTreeClassifier()
params = {
'criterion':['gini','entropy'],
'max_depth':[2,3,4,5,None]  
}
grid = GridSearchCV(dtc,params)
y_pred = grid.fit(X_train,y_train).predict(X_test)
print("Algorithm: Decision Tree (without scaler) \nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print("Acuracia: %f " % (accuracy_score(y_test,y_pred)))
print("F1-score: %f" % (f1_score(y_test, y_pred, average = 'macro')))
print("Precisao: %f\n" % (precision_score(y_test, y_pred, average = 'macro')))
print(classification_report(y_test,y_pred))
print("Melhores parametros:",grid.best_params_,'\n')


clf = make_pipeline(StandardScaler(),dtc)
params = {
'standardscaler__with_mean':[True,False],
'standardscaler__with_std':[False,True],
'decisiontreeclassifier__criterion':['gini','entropy'],
'decisiontreeclassifier__max_depth':[2,3,4,5,None]
}
grid3 = GridSearchCV(clf,params)
y_pred = grid3.fit(X_train,y_train).predict(X_test)
print("Algorithm: Decision Tree \nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print("Acuracia: %f" % (accuracy_score(y_test,y_pred)))
model_accuracy.append(accuracy_score(y_test,y_pred))
print("F1-score: %f" % (f1_score(y_test, y_pred, average = 'macro')))
model_f1_score.append(f1_score(y_test, y_pred, average = 'macro'))
print("Precisao: %f" % (precision_score(y_test, y_pred, average = 'macro')))
model_precision.append(precision_score(y_test, y_pred, average = 'macro'))
print("Melhores parametros:",grid3.best_params_,'\n')
#y_proba = dtc.predict_proba(X_test)
#print(y_proba)
print(classification_report(y_test,y_pred))
#roc_curves.append(roc_curve(y_test,grid3.predict_proba(X_test)[:,1]))


knn = KNeighborsClassifier(6)
params = {
'n_neighbors':[3,4,5,6,7,8],
'weights':['uniform','distance']
}
grid = GridSearchCV(knn,params)
y_pred = grid.fit(X_train,y_train).predict(X_test)
print("Algorithm: 6-Nearest neighbors(without scaler) \nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print("Acuracia: %f " % (accuracy_score(y_test,y_pred)))
print("F1-score: %f" % (f1_score(y_test, y_pred, average = 'macro')))
print("Precisao: %f\n" % (precision_score(y_test, y_pred, average = 'macro')))
print(classification_report(y_test,y_pred))
print("Melhores parametros:",grid.best_params_,'\n')


clf = make_pipeline(StandardScaler(),KNeighborsClassifier())
params = {
'standardscaler__with_mean':[True,False],
'standardscaler__with_std':[False,True],
'kneighborsclassifier__n_neighbors':[3,4,5,6,7,8],
'kneighborsclassifier__weights':['uniform','distance']
}
grid4 = GridSearchCV(clf,params)
y_pred = grid4.fit(X_train,y_train).predict(X_test)
print("Algorithm: 6-Nearest neighbors \nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print("Acuracia: %f" % (accuracy_score(y_test,y_pred)))
model_accuracy.append(accuracy_score(y_test,y_pred))
print("F1-score: %f" % (f1_score(y_test, y_pred, average = 'macro')))
model_f1_score.append(f1_score(y_test, y_pred, average = 'macro'))
print("Precisao: %f" % (precision_score(y_test, y_pred, average = 'macro')))
model_precision.append(precision_score(y_test, y_pred, average = 'macro'))
print("Melhores parametros:",grid4.best_params_,'\n')
print(classification_report(y_test,y_pred))
roc_curves.append(roc_curve(y_test,grid4.predict_proba(X_test)[:,1]))


svm = svm.SVC()
params = {
  'gamma':['scale','auto'],
  'kernel':['rbf','sigmoid'],
  'C':[1,10,100,1000]
}
grid = GridSearchCV(svm,params)
y_pred = grid.fit(X_train,y_train).predict(X_test)
print("Algorithm: Support Vector Machine (SVC without scaler)\nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print("Acuracia: %f " % (accuracy_score(y_test,y_pred)))
print("F1-score: %f" % (f1_score(y_test, y_pred, average = 'macro')))
print("Precisao: %f\n" % (precision_score(y_test, y_pred, average = 'macro', zero_division = 1)))
print("Melhores parametros:",grid.best_params_,'\n')

print(classification_report(y_test,y_pred))


clf = make_pipeline(StandardScaler(), SVC(probability=True))
params = {
'standardscaler__with_mean':[True,False],
'standardscaler__with_std':[False,True],
'svc__gamma':['scale','auto'],
'svc__kernel':['rbf','sigmoid'],
'svc__C':[0.01,0.1,1,10,100,1000]
}
grid5 = GridSearchCV(clf,params)
y_pred = grid5.fit(X_train, y_train).predict(X_test)
print("Algorithm: Support Vector Classification \nNumber of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum())) 
print("Acuracia: %f" % (accuracy_score(y_test,y_pred)))
model_accuracy.append(accuracy_score(y_test,y_pred))
print("F1-score: %f" % (f1_score(y_test, y_pred, average = 'macro')))
model_f1_score.append(f1_score(y_test, y_pred, average = 'macro'))
print("Precisao: %f" % (precision_score(y_test, y_pred, average = 'macro')))
model_precision.append(precision_score(y_test, y_pred, average = 'macro'))
print("Melhores parametros:",grid5.best_params_,'\n')
print(classification_report(y_test,y_pred))
roc_curves.append(roc_curve(y_test,grid5.predict_proba(X_test)[:,1]))


#para montar os gráficos de barra dos algoritmos para precisão, acurácia e f-score

results['model_labels'] = ['Gauss NB','Compl NB','Decision Tree','KNN','SVC']
results['accuracy'] = model_accuracy
results['precision'] = model_precision
results['f1'] = model_f1_score
sns.set_style('dark')
results = results.melt(id_vars='model_labels',var_name='results')
sns.catplot(x = 'model_labels', y='value', hue = 'results',data=results, kind='bar')
plt.savefig("results.png",format='png')
plt.clf()
models = ['Gauss NB','Compl NB','KNN','SVC']
colors = ['red','blue','orange','green']
plt.figure()
plt.xlabel("Taxa FP")
plt.ylabel("Taxa VP")
for i in range(4):
   fp = roc_curves[i][0]
   tp = roc_curves[i][1]
   print(models[i]," AUC:",auc(fp,tp))
   plt.plot(fp,tp,color=colors[i],label=models[i])

#pegar média e desvio padrão de todos algoritmos
#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
#plotar diferentes gráficos de cada algoritmos
#acuracia, precisão, f1 score,
plt.legend(loc='best')
plt.savefig('roc.png')
#plt.show()