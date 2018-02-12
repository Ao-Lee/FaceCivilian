'''
compute similarity and evaluate result
'''
import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN
from sklearn import metrics


path_output_array = 'E:\\DM\\Faces\\Data\\PCD\\array.npy'
path_output_person_idx = 'E:\\DM\\Faces\\Data\\PCD\\person_idx.npy'

threshold = 0.6
eps=0.65
min_samples=4

if __name__=='__main__':
    embs = np.load(path_output_array)
    labels_true = np.load(path_output_person_idx)
    # embs = StandardScaler().fit_transform(embs)
    embs = normalize(embs)
    
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(embs)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    for idx in range(len(set(labels_true))):
        mask = (labels==idx)
        total = np.sum(mask)
        if total==0:
            continue
        print(labels_true[mask])
    
    print('min_samples: %d' % min_samples)
    print('eps: %0.2f' % eps)
    print('Estimated number of clusters: %d' % n_clusters_)
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
   
    
    
    
    