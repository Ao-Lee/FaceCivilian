import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, cdist, squareform

import warnings
warnings.filterwarnings("ignore")

path_output_array = 'E:\\DM\\Faces\\Data\\PCD\\array.npy'
path_output_person_idx = 'E:\\DM\\Faces\\Data\\PCD\\person_idx.npy'

eps=0.65
min_samples=4

# manually tune DBSCAN
# two params are important: eps and min_samples
'''
主要API
'''
def TuneDBSCAN():
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
   
   
# use DBSCAN to group similar face together
# inputs: a ndarray of shape (P, D)
# where P = number of images, D = dim of network output (4096 by default)
# output: a list of ndarray of shape (Pi, D)
# where Pi is size of cluster i
def _Clustering(embs, info):
    embs = normalize(embs)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(embs)
    labels = db.labels_
    clusters = []
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    for idx in range(n_clusters_):
        mask = labels==idx
        cluster = (embs[mask,:], info[mask])
        clusters.append(cluster)
    return clusters
    
# returns 5 people who are most similar to each other
# notice that the complexity of optimal solution 
# is non-polinominal, so I use greedy method instead
# inputs: a numpy array of shape (P, D)
# if P <= 5, return input
def _GreedySimilar(embs, info, num=5):
    total = embs.shape[0]
    if total<=num:
        return embs, info
    dist = pdist(embs, metric='euclidean')
    dist = squareform(dist)
    # select the first face who is most similar to others
    idx_selected = dist.sum(axis=0).argmin()
    included = embs[idx_selected,:].reshape(1, -1)
    reverse_mask = ( np.arange(total)!=idx_selected )
    excluded = embs[reverse_mask,:]
    info_included = []
    info_included.append(info[idx_selected])
    info_excluded = info[reverse_mask]
    for _ in range(num-1):
        dist = cdist(included, excluded)
        idx_selected = dist.sum(axis=0).argmin()
        selected = excluded[idx_selected,:].reshape(1, -1)
        included = np.concatenate([included,selected],axis=0)
        reverse_mask = ( np.arange(excluded.shape[0])!=idx_selected )
        excluded = excluded[reverse_mask,:]
        info_included.append(info_excluded[idx_selected])
        info_excluded = info_excluded[reverse_mask]

    info_included = np.array(info_included)
    return (included, info_included)
    
'''
主要API
'''
def Organize(embs, info, debug=False):
    # embs = np.concatenate(list_embs, axis=0)
    clusters = _Clustering(embs, info)
    clusters = [_GreedySimilar(cluster, current_info) for (cluster, current_info) in clusters]
    if debug:
        for cluster, current_info in clusters:
            # print(cluster.shape)
            print(current_info)
    return clusters
    
if __name__=='__main__':
    embs = np.load(path_output_array)
    info = np.load(path_output_person_idx)
    clusters = Organize(embs, info.astype(str))
    
    # random customer
    customer = np.random.random_integers(low=0, high=len(info))
    customer_info = str(info[customer])
    customer_emb = embs[customer,:].reshape(1, -1)

    best_info = None
    best_dist = np.inf
    for cluster, current_info in clusters:
        dist = np.sum(cdist(customer_emb, cluster))
        if dist < best_dist:
            best_dist = dist
            best_info = current_info
            
    print('客户ID是: ' + customer_info)
    print('最相似的人是: ' + str(best_info))
        

    