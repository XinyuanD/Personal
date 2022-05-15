# Xinyuan Duan
# 9 March 2022
# EPS TICS:AIML
#
# Predicts wine quality using the UCI wine dataset

# import libraries
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# read in white wine data
wine_white = pd.DataFrame(pd.read_csv("winequality-white.csv", sep=";"))
print(wine_white.head())
print(wine_white[200:400])

# pre-processing visualization
sns.set (font_scale=1.1)
sns.set_style('whitegrid')
grid = sns.pairplot(data=wine_white, vars=wine_white.columns[0:11], hue='quality')
plt.show()

# X is feature data, y is target data
X = pd.DataFrame(wine_white.drop(['quality'], axis=1).values)
y = wine_white['quality'].values
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# I commented this and the next code block out because feature reduction does not work well
# feature reduction with PCA
# pca = PCA(n_components=2, random_state=11)
# pca.fit(X)
# X = pca.transform(X)

# feature reduction with TSNE
# tsne = TSNE(n_components=2, random_state=11)
# X = tsne.fit_transform(X)

# split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# I commented this out because the accuracy for each odd k from 1 to 30 is pretty much the same,
# k=1 has the highest accuracy but I think k=1 is too few neighbors to look at
# for k in range(1,30,):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(X=X_train, y=y_train)
#     predicted = knn.predict(X=X_test)
#     expected = y_test
#     print("k is {}, accuracy is {}".format(k, metrics.accuracy_score(expected, predicted)))

# creating and training a knn model, then predict it on X_test
knn = KNeighborsClassifier()
knn.fit(X=X_train, y=y_train)
predicted = knn.predict(X=X_test)
expected = y_test
print(metrics.accuracy_score(expected, predicted))

# I commented this out because I tried this and this does not work well
# creating and training a linear regression model
# lin_reg = LinearRegression()
# lin_reg.fit(X=X_train, y=y_train)
# predicted = lin_reg.predict(X_test)
# expected = y_test
# print(metrics.r2_score(expected, predicted))

# train kmeans cluster model
kmeans = KMeans(n_clusters=7, random_state=11)
kmeans.fit(X)

# visualize the data by using PCA feature reduction
pca = PCA(n_components=2, random_state=11)
pca.fit(X)
wine_pca = pca.transform(X)
print(wine_pca.shape)
wine_pca_df = pd.DataFrame(wine_pca, columns=['Component1', 'Component2'])
wine_pca_df['quality'] = y
axes = sns.scatterplot(data=wine_pca_df, x='Component1', y='Component2', 
                    hue='quality', legend='brief', palette='cool')
wine_center = pca.transform(kmeans.cluster_centers_)
dots = plt.scatter(wine_center[:,0], wine_center[:,1], s=100, c='k')
plt.show()

# I commented this out because this does not work well and I already have the PCA visualization
# visualize the data by using TSNE feature reduction
# tsne = TSNE(n_components=2, random_state=11)
# reduced_data = tsne.fit_transform(X)
# dots = plt.scatter(reduced_data[:,0], reduced_data[:,1], 
#                     c=y, cmap=plt.cm.get_cmap('nipy_spectral_r', 7))
# colorbar = plt.colorbar(dots)
# plt.show()