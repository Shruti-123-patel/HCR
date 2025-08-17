import os
import cv2
from os import listdir
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import preProcess
from PIL import Image as im
from K_mean import createModel
from Model_NN import NN_model
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from vggNet import vggNet
from numpy import save,load,asarray
from test import evaluate
sizeOfImage = 64

# preprocessing of image 
# normalized_array = asarray(preProcess(sizeOfImage))
# print(normalized_array.shape)
# save('augmentedData.npy', normalized_array)

# import images name with its lable
# df = pd.read_csv("../Dataset/english.csv")
df = pd.read_excel("augmentedData.xls")

# split the data in training and testing sets
X = load('augmentedData.npy')
Y = df['label']

le = preprocessing.LabelEncoder()
le.fit(Y)

Y= le.transform(Y)

onehotEncoding = OneHotEncoder()
Y = np.array(Y).reshape(-1,1)
Y = onehotEncoding.fit_transform(Y).toarray()
# y_print = pd.DataFrame(Y[2202:2252,])
# Y_mapping = dict(zip(onehotEncoding.transform(le.classes_.reshape(-1,1)),le.classes_))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42,stratify = Y,shuffle=True)
print(Y)
# model = createModel(X_train, X_test, y_train, y_test,sizeOfImage)
vggNet(X_train, X_test, y_train, y_test,sizeOfImage)
# evaluate(X_test,y_test)