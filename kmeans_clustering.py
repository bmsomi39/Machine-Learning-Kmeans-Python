# K-Means Clustering

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the mall dataset with pandas

dataset = pd.read_csv('dataBoth.csv')
X = dataset.iloc[:,[1,2]].values #(albania,13.5,10)

# Using the elbow method to find the optimal number of clusters

from sklearn.cluster import KMeans
wcss =[]
for i in range (1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter =300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)



# Applying KMeans to the dataset with the optimal number of cluster
k_clusters=int(input("Enter number of k(clusters)?"))

print(k_clusters)



#change n_clusters= 2 : n_clusters= k_clusters
kmeans=KMeans(n_clusters= k_clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_Kmeans = kmeans.fit_predict(X)




# Visualising the clusters


plt.scatter(X[y_Kmeans == 0, 0], X[y_Kmeans == 0,1],s = 100, c='red', label = 'Cluster 1')



plt.scatter(X[y_Kmeans == 1, 0], X[y_Kmeans == 1,1],s = 100, c='blue', label = 'Cluster 2')

plt.scatter(X[y_Kmeans == 2, 0], X[y_Kmeans == 2,1],s = 100, c='green', label = 'Cluster 3')

plt.scatter(X[y_Kmeans == 3, 0], X[y_Kmeans == 3,1],s = 100, c='cyan', label = 'Cluster 4')

plt.scatter(X[y_Kmeans == 4, 0], X[y_Kmeans == 4,1],s = 100, c='magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')
    
plt.title('Clusters')
plt.xlabel('BirthRate(Per1000)')
plt.ylabel('LifeExpectancy')
plt.legend()
plt.show()










#Output metadata

estimator = KMeans(n_clusters=k_clusters)
estimator.fit(X)

print()
print('List of indices in each cluster:')
list_indece = {i: np.where(estimator.labels_ == i)[0] for i in range(estimator.n_clusters)} #get the indices of points for each cluster
print(list_indece)



print()
print('Number of data points in each cluster:  ')
from collections import Counter, defaultdict
print(Counter(estimator.labels_))


    

print()
print('List of countries in each cluster: ')

print()    
print("*************")
for k, v in list_indece.items():
    country = []
    
    for i,j in enumerate(dataset.iloc[:,0]):
        for c in v:
            if c == i+1:
                country.append(j) 
                
    print(f"__________________cluster{k}_____________________________")
    print(country)


print()
print('List of mean of Life Expectancy & Birthdate for each cluster:')
for k, v in list_indece.items():
    
    print("Cluster",k, ":", v.mean())


