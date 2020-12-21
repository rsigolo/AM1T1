import csv
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

with open('/home/augusto/ufscar/ENPE Bloco B/AM/heart_failure_clinical_records_dataset.csv', 'r') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  csv_reader = list(csv_reader)
  csv_reader = np.array(csv_reader[1:300]).astype("float")

X=csv_reader[1:300, 0:12]
X = StandardScaler().fit_transform(X)
Y=csv_reader[1:300, 12]
fig = plt.figure(figsize=(10,10))

pca = PCA(n_components=2)

pc = pca.fit_transform(X)

df = pd.DataFrame(data=pc,columns=['principal component 1','principal component 2'])
df['y'] = Y
seaborn.scatterplot(x='principal component 1',y='principal component 2',hue='y',palette='viridis',data=df,alpha=0.8)
#plt.savefig('pca.png')
#plt.show()
