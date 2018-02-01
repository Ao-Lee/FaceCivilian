'''
compute similarity and evaluate result
'''
import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components

path_output_array = 'E:\\DM\\Faces\\Data\\PCD\\array.npy'
path_output_person_idx = 'E:\\DM\\Faces\\Data\\PCD\\person_idx.npy'

threshold = 0.6

def DistanceEuclidean(X, Y):
    X = X.reshape(1, -1)
    Y = Y.reshape(1, -1)
    diff = (normalize(X) - normalize(Y))
    return (diff**2).sum()

def GetDistance(embeddings):
    dist = pdist(embeddings, metric='euclidean')
    dist = squareform(dist)
    dist = dist**2
    return dist
    
if __name__=='__main__':
    embs = np.load(path_output_array)
    embs = normalize(embs)
    persons_idx = np.load(path_output_person_idx)
    dist = GetDistance(embs)
    dist = dist < threshold
    n_components, labels = connected_components(dist, directed=False)
    for idx in range(n_components):
        mask = (labels==idx)
        total = np.sum(mask)
        if total<=3:
            labels[mask]=-1

    # compare ground truth lables with labels computed from connected component algorithm
    for idx in range(n_components):
        mask = (labels==idx)
        total = np.sum(mask)
        if total==0:
            continue
        print(persons_idx[mask])
        
    
    
    
    