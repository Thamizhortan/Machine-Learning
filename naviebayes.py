
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from sklearn import tree
import graphviz


# Load data
data = pd.read_csv('Data/data.csv', delimiter=',')

# Head method show first 5 rows of data
print(data.head())

# Drop unused columns
columns = ['Unnamed: 32', 'id', 'diagnosis']

# Convert strings -> integers
d = {'M': 0, 'B': 1}

# Define features and labels
y = data['diagnosis'].map(d)
X = data.drop(columns, axis=1)

# Plot number of M - malignant and B - benign cancer

ax = sns.countplot(y, label="Count", palette="muted")
B, M = y.value_counts()
plt.savefig('Plots/count.png')
print('Number of benign cancer: ', B)
print('Number of malignant cancer: ', M)


# Split dataset into training (80%) and test (20%) set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize data
X_train_N = (X_train-X_train.mean())/(X_train.max()-X_train.min())
X_test_N = (X_test-X_train.mean())/(X_test.max()-X_test.min())

####### PCA ######


# PCA without std
pca = PCA()
pca.fit(X)
print(pca.explained_variance_ratio_)
plt.figure()
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('number of components')
plt.ylabel('% of variance')
plt.title('PCA without Std')
plt.savefig('Plots/pcavariancewithoutstd.png')

# PCA with std
pca = PCA(n_components=6)
X_std = StandardScaler().fit_transform(X)
pca.fit(X_std)
print(pca.explained_variance_ratio_)
plt.figure()
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('number of components')
plt.ylabel('% of variance')
plt.title('PCA with Std')
#plt.savefig('Plots/pcavariancewithstd.png')


###### Naive Bayes Classifier ######

gnb = GaussianNB()
gnbclf = gnb.fit(X_train_N, y_train)
gnbp = Pipeline([
    ('reduce_dim', pca),
    ('gnb', gnb)
])
gnb_score = cross_val_score(gnb, X, y, cv=10, scoring='accuracy')
print("Gaussian Naive Bayes accuracy: %0.2f (+/- %0.2f)" % (gnb_score.mean(), gnb_score.std() * 2))
