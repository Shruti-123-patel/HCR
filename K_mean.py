from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from PIL import Image as im


def createModel(X_train,X_test,Y_train,Y_test):
    kmeans = KMeans(n_clusters=62, random_state=56,n_init=62)
    X_1d = []
    for a in X_train:
        X_1d.append(a.flatten())
    kmeans.fit(X_1d)
    print(kmeans.cluster_centers_)
    fig, ax = plt.subplots(31, 2, figsize = (10,5))
    centers = kmeans.cluster_centers_.reshape(62,28,28)
    print(X_1d[0])
    