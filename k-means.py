#Import Module
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Import File CSV
file_dataset = str(input('Masukan Nama File Dataset : '))
data = pd.read_csv(file_dataset)

#Menentukan Kolom yang Digunakan dan Jumlah Cluster
features = data[['umur', 'bmi']]
i = int(input("Masukan Jumlah Cluster : "))

#Mengcluster
kmeans = KMeans(n_clusters=i)
data['Cluster_KMeans'] = kmeans.fit_predict(features)

#Memvisualisasikan Hasil Clustering
plt.scatter(data['umur'], data['bmi'], c=data['Cluster_KMeans'], cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=30, c='black', label='Centroids')
plt.xlabel('umur')
plt.ylabel('bmi')
plt.title(f'KMeans Clustering')
plt.legend()
plt.show()