import numpy as np
from scipy.sparse import csr_matrix, issparse
from time import time
import warnings
from sklearn.neighbors import NearestNeighbors


def S(k, t,Y_train,Y_test):
    # Assuming data is loaded from somewhere, e.g., from a file
    # Y_train and Y_test should be loaded or passed as arguments to the function
    # Y_train = np.load('path_to_Y_train.npy')
    # Y_test = np.load('path_to_Y_test.npy')

    options = {
        'Metric': 'Euclidean',#Cosine
        'WeightMode': 'HeatKernel',#Binary/Cosine
        'NeighborMode': 'KNN',
        'k': k,
        't': t
    }

    ntrn = Y_train.shape[0]
    ntest = Y_test.shape[0]
    
    temp2 = np.zeros((ntrn, ntest))
    for i in range(ntest):
        fmri = np.vstack([Y_train, Y_test[i, :]])
        temp,elapsed_time = constructW(fmri, options)
        temp2[:, i] = temp[:-1, -1]

    S = np.zeros((ntrn, ntest))
    for i in range(ntest):
        sorted_indices = np.argsort(-temp2[:, i])
        selectidx = sorted_indices[1:k+1]  # Skip the first (highest) value like MATLAB idx(2:k+1,i)
        S[selectidx, i] = -np.sort(-temp2[:, i])[1:k+1]

    return S


def EuDist2(fea_a, fea_b=None, bSqrt=True):
    """
    computes the Euclidean distance between two sets of feature vectors fea_a and fea_b.
    If fea_b is not provided, it computes the pairwise distances within fea_a.

    Parameters:
        fea_a: ndarray of shape (nSample_a, nFeature)
        fea_b: ndarray of shape (nSample_b, nFeature), optional
        bSqrt: bool, optional, whether to take the square root of the result

    Returns:
        D: ndarray of shape (nSample_a, nSample_a) or (nSample_a, nSample_b)
    """

    if fea_b is None:
        aa = np.sum(fea_a * fea_a, axis=1)
        ab = np.dot(fea_a, fea_a.T)

        if issparse(aa):
            aa = aa.toarray().flatten()

        D = aa[:, np.newaxis] + aa - 2*ab
        D[D < 0] = 0

        if bSqrt:
            D = np.sqrt(D)

        D = np.maximum(D, D.T)

    else:
        aa = np.sum(fea_a * fea_a, axis=1)
        bb = np.sum(fea_b * fea_b, axis=1)
        ab = np.dot(fea_a, fea_b.T)

        if issparse(aa):
            aa = aa.toarray().flatten()
            bb = bb.toarray().flatten()

        D = aa[:, np.newaxis] + bb - 2*ab
        D[D < 0] = 0

        if bSqrt:
            D = np.sqrt(D)

    return D

# Placeholder for LLE_Matrix function
def LLE_Matrix(data, k, regLLE):
    # Implement or convert this function from MATLAB if needed
    W = None  # Placeholder
    M = None  # Placeholder
    return W, M



def constructW2(X, options=None):
    # Example usage:
    # X = np.array([[1, 2], [3, 4], [2, 2], [8, 9]])
    # options = {
    #     'NeighborMode': 'KNN',
    #     'WeightMode': 'Binary',
    #     'k': 2
    # }
    # W = constructW(X, options)
    # print(W)
    if options is None:
        options = {}

    # Set default options
    metric = options.get('Metric', 'Cosine')
    neighbor_mode = options.get('NeighborMode', 'KNN')
    weight_mode = options.get('WeightMode', 'Binary')
    k = options.get('k', 5)  # Default number of neighbors

    num_samples = X.shape[0]
    W = np.zeros((num_samples, num_samples))

    # If using supervised mode
    if neighbor_mode == 'Supervised':
        labels = options.get('label')
        if weight_mode == 'Binary':
            for i in range(num_samples):
                for j in range(num_samples):
                    if labels[i] == labels[j]:
                        W[i, j] = 1
        # Add other weight modes here if needed

    # If using KNN
    elif neighbor_mode == 'KNN':
        if k > 0:
            nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric.lower()).fit(X)
            distances, indices = nbrs.kneighbors(X)
            for i in range(num_samples):
                if weight_mode == 'Binary':
                    W[i, indices[i, 1:]] = 1
                elif weight_mode == 'HeatKernel':
                    W[i, indices[i, 1:]] = np.exp(-distances[i, 1:]**2)
                # Add other weight modes here if needed

    # Normalize the weight matrix (optional)
    # sum_W = np.sum(W, axis=1)
    # W = W / sum_W[:, np.newaxis]

    return W




def constructW(fea, options=None):
    # Melengkapi opsi di options
    if options is None:
        options = {}

    if "LLE" in options and options["LLE"]:
        start_time = time()
        W, M = LLE_Matrix(fea.T, options["k"], options["regLLE"])
        elapsed_time = time() - start_time
        return W, elapsed_time, M
    
    options.setdefault("Metric", "Cosine")
    
    if options["Metric"].lower() not in ["euclidean", "cosine"]:
        raise ValueError("Metric does not exist!")
    
    if options["Metric"].lower() == "cosine" and "bNormalized" not in options:
        options["bNormalized"] = 0
    
    options.setdefault("NeighborMode", "KNN")
    
    if options["NeighborMode"].lower() not in ["knn", "supervised"]:
        raise ValueError("NeighborMode does not exist!")
    
    if options["NeighborMode"].lower() == "supervised":
        if "bLDA" not in options:
            options["bLDA"] = 0
        if options["bLDA"]:
            options["bSelfConnected"] = 1
        if "k" not in options:
            options["k"] = 0
        if "gnd" not in options:
            raise ValueError("Label(gnd) should be provided under 'Supervised' NeighborMode!")
        if fea.shape[0] != len(options["gnd"]):
            raise ValueError("gnd doesn't match with fea!")
    
    options.setdefault("WeightMode", "Binary")
    
    if options["WeightMode"].lower() not in ["binary", "heatkernel", "cosine"]:
        raise ValueError("WeightMode does not exist!")
    
    if options["WeightMode"].lower() == "heatkernel" and options["Metric"].lower() != "euclidean":
        warnings.warn("'HeatKernel' WeightMode should be used under 'Euclidean' Metric!")
        options["Metric"] = "euclidean"
        if "t" not in options:
            options["t"] = 1
    
    if options["WeightMode"].lower() == "cosine" and options["Metric"].lower() != "cosine":
        warnings.warn("'Cosine' WeightMode should be used under 'Cosine' Metric!")
        options["Metric"] = "cosine"
        if "bNormalized" not in options:
            options["bNormalized"] = 0
    
    options.setdefault("bSelfConnected", 1)
    
    start_time = time()
    
    if "gnd" in options:
        nSmp = len(options["gnd"])
    else:
        nSmp = fea.shape[0]
    
    maxM = 62500000
    BlockSize = maxM // (nSmp * 3)
    print(options)
    # Disini mulai komputasi dari W nya
    #Depending on the chosen options, the function computes W in different ways:
    #If 'Supervised' mode is used, then W is constructed based on class labels (gnd field in options). 
    #Points from the same class can be connected by binary weights, weights based on the heat kernel, or cosine similarity.
    #If 'KNN' mode is used, the function computes the k-nearest neighbors for each point based on the specified metric. 
    #Weights are then assigned to these connections based on the WeightMode.
    
    #Finally, the function returns the weight matrix W, the time taken (elapse), and in some cases, the matrix M (used for LLE).

    # You will need to fill in the logic here as per your needs
    # For Supervised #
    if options["NeighborMode"].lower() == "supervised":
        print("masuk mode supervised")
        Label = np.unique(options["gnd"])
        nLabel = len(Label)

        if options["bLDA"]:
            G = np.zeros((fea.shape[0], fea.shape[0]))
            for idx in Label:
                classIdx = options["gnd"] == idx
                G[classIdx, classIdx] = 1 / np.sum(classIdx)
            W = csr_matrix(G)
            elapse = time() - start_time
            return W, elapse

        if options["WeightMode"].lower() == "binary":
            print("Rest of the logic for 'binary' mode")
            G = np.zeros((nSmp*(options["k"]+1), 3))
            idNow = 0
            for idx in Label:
                classIdx = np.where(options["gnd"] == idx)[0]
                D = EuDist2(fea[classIdx], bSqrt=False)
                idx_sorted = np.argsort(D, axis=1)
                idx_selected = idx_sorted[:, :options["k"]+1]
                
                nSmpClass = len(classIdx) * (options["k"]+1)
                G[idNow:idNow+nSmpClass, 0] = np.repeat(classIdx, options["k"]+1)
                G[idNow:idNow+nSmpClass, 1] = classIdx[idx_selected.ravel()]
                G[idNow:idNow+nSmpClass, 2] = 1
                idNow += nSmpClass

            G = csr_matrix((G[:, 2], (G[:, 0].astype(int), G[:, 1].astype(int))), shape=(nSmp, nSmp))
            G = G.maximum(G.T)

            if not options["bSelfConnected"]:
                G.setdiag(0)

            W = G.copy()
            
        elif options["WeightMode"].lower() == "heatkernel":
            print("Rest of the logic for 'heatkernel' mode")
            G = np.zeros((nSmp*(options["k"]+1), 3))
            idNow = 0
            for idx in Label:
                classIdx = np.where(options["gnd"] == idx)[0]
                D = EuDist2(fea[classIdx], bSqrt=False)
                dump, idx_sorted = np.sort(D, axis=1), np.argsort(D, axis=1)
                idx_selected = idx_sorted[:, :options["k"]+1]
                dump_selected = dump[:, :options["k"]+1]
                
                heat_kernel_weight = np.exp(-dump_selected / (2 * options["t"]**2))
                
                nSmpClass = len(classIdx) * (options["k"]+1)
                G[idNow:idNow+nSmpClass, 0] = np.repeat(classIdx, options["k"]+1)
                G[idNow:idNow+nSmpClass, 1] = classIdx[idx_selected.ravel()]
                G[idNow:idNow+nSmpClass, 2] = heat_kernel_weight.ravel()
                idNow += nSmpClass

            G = csr_matrix((G[:, 2], (G[:, 0].astype(int), G[:, 1].astype(int))), shape=(nSmp, nSmp))
            G = G.maximum(G.T)

            if not options["bSelfConnected"]:
                G.setdiag(0)

            W = G.copy()

        elif options["WeightMode"].lower() == "cosine":
            # Normalize the data if required
            # if not options["bNormalized"]:
            #     feaNorm = np.linalg.norm(fea, axis=1)
            #     fea = fea / np.maximum(feaNorm[:, np.newaxis], 1e-12)
            if not options.get("bNormalized", False):
                fea_norm = np.linalg.norm(fea, axis=1, keepdims=True)
                fea = np.divide(fea, fea_norm, where=fea_norm!=0)

            G = np.zeros((nSmp*(options["k"]+1), 3))
            idNow = 0
            for idx in Label:
                classIdx = np.where(options["gnd"] == idx)[0]
                D = fea[classIdx].dot(fea[classIdx].T)
                
                # We're looking for highest cosine similarity, so we sort in descending order
                dump, idx_sorted = np.sort(-D, axis=1), np.argsort(-D, axis=1)
                idx_selected = idx_sorted[:, :options["k"]+1]
                dump_selected = -dump[:, :options["k"]+1]
                
                nSmpClass = len(classIdx) * (options["k"]+1)
                G[idNow:idNow+nSmpClass, 0] = np.repeat(classIdx, options["k"]+1)
                G[idNow:idNow+nSmpClass, 1] = classIdx[idx_selected.ravel()]
                G[idNow:idNow+nSmpClass, 2] = dump_selected.ravel()
                idNow += nSmpClass

            G = csr_matrix((G[:, 2], (G[:, 0].astype(int), G[:, 1].astype(int))), shape=(nSmp, nSmp))
            G = G.maximum(G.T)

            if not options["bSelfConnected"]:
                G.setdiag(0)

            W = G.copy()

        else:
            raise ValueError("WeightMode does not exist!")
    # For KNN #
    bBinary = options.get('WeightMode') == 'Binary'
    bNormalized = options.get('bNormalized', False)
    bSelfConnected = options.get('bSelfConnected', True)
    if options["NeighborMode"].lower() == "knn" and options["k"] > 0:
        print("masuk mode KNN")
        k = options.get('k')
        G = np.zeros((nSmp*(k+1), 3))
        for i in range(1, int(np.ceil(nSmp/BlockSize)) + 1):
            if i == int(np.ceil(nSmp/BlockSize)):
                smpIdx = np.arange((i-1)*BlockSize, nSmp)
            else:
                smpIdx = np.arange((i-1)*BlockSize, i*BlockSize)
                
            if options.get('Metric') == 'Euclidean':
                dist = EuDist2(fea[smpIdx], fea, bSqrt=False)
                idx = np.argsort(dist, axis=1)[:, :k+1]
                dump = np.sort(dist, axis=1)[:, :k+1]
                if not bBinary:
                    dump = np.exp(-dump / (2 * options.get('t')**2))
                start_idx = (i-1)*BlockSize*(k+1)
                end_idx = start_idx + len(smpIdx)*(k+1)
                G[start_idx:end_idx] = np.column_stack((
                    np.repeat(smpIdx, k+1),
                    idx.ravel(),
                    dump.ravel()
                ))
            else:
                # Here, assume cosine similarity for non-euclidean
                if not bNormalized:
                    feaNorm = np.linalg.norm(fea, axis=1)
                    for i in range(nSmp):
                        fea[i] = fea[i] / max(1e-12, feaNorm[i])
                dist = np.dot(fea[smpIdx], fea.T)
                idx = np.argsort(-dist, axis=1)[:, :k+1]
                dump = -np.sort(-dist, axis=1)[:, :k+1]
                start_idx = (i-1)*BlockSize*(k+1)
                end_idx = start_idx + len(smpIdx)*(k+1)
                G[start_idx:end_idx] = np.column_stack((
                    np.repeat(smpIdx, k+1),
                    idx.ravel(),
                    dump.ravel()
                ))
        W = csr_matrix((G[:,2], (G[:,0].astype(int), G[:,1].astype(int))), shape=(nSmp, nSmp))
        
        if bBinary:
            W.data[:] = 1
        
        if not bSelfConnected:
            W.setdiag(0)
        
        W = W.maximum(W.transpose()).tocsr()
        #return W


    #return part
    elapsed_time = time() - start_time
    return W.toarray(), elapsed_time  # M is not defined in the provided code.

# Usage
# fea = ...  # Some numpy array or data
# options = {"k": 5, "Metric": "euclidean", "NeighborMode": "knn", ...}
# W, elapsed_time = constructW(fea, options)