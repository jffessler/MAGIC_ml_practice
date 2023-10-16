import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import copy
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import LinearRegression
import csv 

dataset_cols = ["bike_count", "hour", "temp", "humidity", "wind", "visibility", "dew_point_temp", "radiation", "rain", "snow", "functional"]
df = pd.read_csv("SeoulBikeData.csv").drop(["Date", "Holiday", "Seasons"], axis=1)
df.columns = dataset_cols
df["functional"] = (df["functional"]=="Yes").astype(int)
df = df[df["hour"]==12]
df = df.drop(["hour"], axis=1)
df = df.drop(["wind","visibility", "functional"], axis=1)
print(df.head())

# for label in df.columns[1:]:
#     plt.scatter(df[label], df["bike_count"])
#     plt.title(label)
#     plt.ylabel("Bike count at noon")
#     plt.xlabel(label)
#     plt.show()