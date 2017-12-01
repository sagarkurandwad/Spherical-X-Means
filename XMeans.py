"""
Created on Thu Jun 19 11:18:50 2017

@author: Sagar Kurandwad
"""

import numpy as np
from numpy import (dot, arccos, clip)
from sklearn.preprocessing import normalize
from spherical_kmeans import SphericalKMeans



def getLabels(Data,Centroids):
    '''
    Return Cluster labels of each data point
    
    Parameters:
    ----------
    Data(array): doc vec array
    Centroids(nested list): List of centroids
    
    Return:
    ------
    (array): Cluster labels for each data point
    '''
    similarity = []
    for clID in range(len(Centroids)):
        similarity.append((dot(Data,np.array(Centroids[clID]))).tolist())
    simArray = np.array(similarity)
    return np.argmax(simArray,axis=0)



def getAngle(V1,V2):
    '''
    Return angle between arrays V1 and V2 in degrees
    
    Parameters:
    ----------
    V1,V2(array): Arrays of doc vecs and/or centroids
    
    Return:
    ------
    (array): Angle array in degrees
    '''
    cos = dot(V1,V2)
    return np.degrees(arccos(clip(cos, -1, 1)))



def sphericalClustering(data,parent,maxClusters):
    '''
    Return centroids and data labels for correct number of clusters after 
    Spherical clustering on data for #clusters = 1 to maxClusters
    
    Parameters:
    ----------
    data(array): doc vectors array
    parent(nested list): parent centroid 
    maxClusters(int): Maximum number of clusters to grid search on
    
    Return:
    ------
    centroids(nested list): cluster centroids
    labels(array): data labels
    '''
    centroids = [np.array(parent)]
    m = data.shape[1]
    n = data.shape[0]
    labels = [np.array([0.0]*n)]
    BIC = []
    
    minClusterSize = maxClusters+1
    
    if n < minClusterSize:
        return centroids[0].tolist(),labels[0]
    else:
        for cl in np.arange(maxClusters)[1:]:
            ssa = 0 # Sum of Squared Angles
            for k in range(cl):
                indices = np.where(labels[cl-1] == float(k))[0].tolist()
                X = data[indices]
                ssa = ssa + np.square(getAngle(X,centroids[cl-1][k])).sum()
            K = cl*(m+2)-1
            BIC.append((ssa+np.log(n)*m*K))

            clModel = SphericalKMeans(n_clusters=cl+1, init='k-means++', n_init=20)
            clModel.fit(data)
            
            centroids.append(normalize(clModel.cluster_centers_,axis=1))
            labels.append(clModel.labels_) 
        
        clusterIndex = BIC.index(min(BIC))
        
        return centroids[clusterIndex].tolist(),labels[clusterIndex]


def XMeans(data,parent,maxClusters):
    '''
    XMeans on data
    
    Parameters:
    ----------
    data(array): data array
    parent(nested list): parent centroids
    maxClusters(int): maximum number of clusters to grid search on
    
    Return:
    ------
    Centroids(nested list): cluster centroids  
    '''
    centroids,labels = sphericalClustering(data,parent,maxClusters)
    
    if len(centroids) == 1:
        Centroids = centroids
    else:
        Centroids = []
        for clID in range(len(centroids)):
            Indices = np.where(labels == float(clID))[0].tolist()
            Centroids.extend(XMeans(data[Indices],[centroids[clID]],maxClusters))
    return Centroids
    

def XMeansTraining(data,maxClusters=10,norm=True):
    '''
    Train XMeans on data
    
    Parameters:
    ----------
    data(array): data array
    maxClusters(int): maximum number of clusters to grid search on
    norm(Boolean): If data is normalized
    
    Return:
    ------
    Centroids(nested list): cluster centroids
    Labels(array): Cluster labels of data points
    '''
    if not norm:
        data = normalize(data,axis=1)
    
    parent = [normalize(np.mean(data,axis=0).reshape(1,-1),axis=1)[0]]
    Centroids = XMeans(data,parent,maxClusters)
    Labels = getLabels(data,Centroids)
    
    return Centroids,Labels

