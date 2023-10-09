import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler


cols = ["fLength","fWidth","fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"]
df = pd.read_csv("magic04.data", names=cols)

df["class"] = (df["class"] == "g").astype(int)
# print(df)
# train_df = df.head()
# print(df["class"].unique())
# print(train_df)

# for label in cols[:-1]:
#     plt.hist(df[df["class"]==1][label], color='blue', label='gamma', alpha = 0.7, density=True)
#     plt.hist(df[df["class"]==0][label], color='red', label='hadron', alpha = 0.7, density=True)
#     plt.title(label)
#     plt.ylabel("Probability")
#     plt.xlabel(label)
#     plt.legend()
#     plt.show()

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

# print(len(train),len(valid),len(test))

def scale_dataset(dataframe, oversample = False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X,y)

    data = np.hstack((X,np.reshape(y,(-1,1))))
    return data,X,y

train,X_train,y_train = scale_dataset(train, oversample=True)
valid,X_valid,y_valid = scale_dataset(valid, oversample=False)
test,X_test,y_test = scale_dataset(test, oversample=False)

###### kNN modeling

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report

# knn_model = KNeighborsClassifier(n_neighbors=5)
# knn_model.fit(X_train,y_train)
# y_pred = knn_model.predict(X_test)
# print(classification_report(y_test,y_pred))

##### Naive Bayes modeling

# from sklearn.naive_bayes import GaussianNB

# nb_model = GaussianNB()
# nb_model = nb_model.fit(X_train,y_train)

# y_pred = nb_model.predict(X_test)
# print(classification_report(y_test,y_pred))


##### Logistic Regression modeling 

# from sklearn.linear_model import LogisticRegression

# lg_model = LogisticRegression()
# lg_model = lg_model.fit(X_train,y_train)

# y_pred = lg_model.predict(X_test)
# print(classification_report(y_test,y_pred))

#### Support Vector Machines (SVM)

# from sklearn.svm import SVC

# svm_model = SVC()
# svm_model = svm_model.fit(X_train,y_train)

# y_pred = svm_model.predict(X_test)
# print(classification_report(y_test,y_pred))

##### Neural Network

import tensorflow as tf

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel("Epoch")
    plt.ylabel("Binary Crossentropy")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape = (10,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
nn_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])

history = nn_model.fit(X_train,y_train,epochs=100,batch_size=32,validation_split=0.2,verbose=0)

plot_loss(history)
plot_accuracy(history)