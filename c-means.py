#Import Module
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz

#Import File CSV
file_dataset = str(input('Masukan Nama File Dataset : '))
data = pd.read_csv(file_dataset)

#Menentukan Kolom yang Digunakan dan Jumlah Cluster
features = data[['umur', 'bmi']]
i = int(input("Masukan Jumlah Cluster : "))

# Metode Algoritma Fuzzy C-Means
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(features.T, i, 2, error=0.005, maxiter=1000)

# Mengambil Datapoint Cluster berdasarkan Derajat Keanggotaan Tertinggi
cluster_membership = np.argmax(u, axis=0)

# Hasilnya ditampilkan dalam plot
fig, ax = plt.subplots()
colors = ['g', 'b', 'c', 'm', 'y', 'k']
for j in range(i):
    ax.scatter(data[cluster_membership == j]['umur'], data[cluster_membership == j]['bmi'],
               c=colors[j], label=f'Cluster {j + 1}', marker='o')
    
# Plot dari Centroid
for i in range(i):
    ax.scatter(cntr[i, 0], cntr[i, 1], marker='o', s=20, linewidths=3, color='red', label=f'Centroid {i + 1}')

#Menampilkan visualisasi dari C-Means
ax.legend()
plt.xlabel('Umur')
plt.ylabel('bmi')
plt.title('C-Means Clustering')
plt.show()