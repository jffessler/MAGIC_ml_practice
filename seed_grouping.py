import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.preprocessing import StandardScaler
# import copy
import seaborn as sns
# import tensorflow as tf
# from sklearn.linear_model import LinearRegression

cols = ["area", "perimeter", "compactness", "length", "width", "asymmetry", "groove", "class"]
df = pd.read_csv("seeds_dataset.txt", names= cols, sep= "\s+")

# print(df.head())

# for i in range(len(cols)-1):
#     for j in range(i+1, len(cols)-1):
#         x_label = cols[i]
#         y_label = cols[j]
#         sns.scatterplot(x=x_label, y=y_label, data=df, hue="class")
#         plt.show()


##### Clustering 

from sklearn.cluster import KMeans

x = "compactness"
y = "asymmetry"
X = df[[x, y]].values
kmeans = KMeans(n_clusters= 3).fit(X)

clusters = kmeans.labels_

# print(clusters)
# print(df["class"].values)

clusters_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1, 1))), columns=[x, y, "class"])

# #K Means classes 
# sns.scatterplot(x= x, y=y, hue="class", data=clusters_df)
# plt.plot()
# plt.show()

# # original classes 
# sns.scatterplot(x= x, y=y, hue="class", data=df)
# plt.plot()
# plt.show()

#Higer Dimensions
X = df[cols[:-1]].values
kmeans = KMeans(n_clusters=3).fit(X)
clusters_df = pd.DataFrame(np.hstack((X, clusters.reshape(-1, 1))), columns=df.columns)

# #K Means classes 
# sns.scatterplot(x= x, y=y, hue="class", data=clusters_df)
# plt.plot()
# plt.show()

# # original classes 
# sns.scatterplot(x= x, y=y, hue="class", data=df)
# plt.plot()
# plt.show()

##### PCA

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
transformed_x   =  pca.fit_transform(X)

# print(X.shape)
# print(transformed_x.shape)

# print(transformed_x[:5])

# plt.scatter(transformed_x[:,0], transformed_x[:,1])
# plt.show()

kmean_pca_df = pd.DataFrame(np.hstack((transformed_x, kmeans.labels_.reshape(-1,1))), columns=["pca1", "pca2", "class"])
truth_pca_df = pd.DataFrame(np.hstack((transformed_x, df["class"].values.reshape(-1,1))), columns=["pca1", "pca2", "class"])

# K Means classes
sns.scatterplot(x= "pca1", y= "pca2", hue="class", data=kmean_pca_df)
plt.plot()
plt.show()

# Truth Classes 

sns.scatterplot(x= "pca1", y= "pca2", hue="class", data=truth_pca_df)
plt.plot()
plt.show()