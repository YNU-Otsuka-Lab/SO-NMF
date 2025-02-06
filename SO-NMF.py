#--------------------------------------------------------------------------------
# Copyright (C) 2025, Yokohama National University, all rights reserved.
# See License.txt in the project root for license information.
#--------------------------------------------------------------------------------

import torch
import numpy as np
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error,mean_absolute_error
# Import Asteris's public code (onmf.py) for ONMF. https://github.com/signofyouth/torch-onmf/blob/main/onmf.py
from onmf import onmf, low_rank_nnpca, _local_opt_w, local_opt_w_single, local_opt_w_batch, binarize, sphere_sample_cartesian

def fix_seed_ONMF(seed): 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Extract an orthogonal basis
def orthogonal_check(basis,new_basis): 
    for i in range(basis.shape[0]):
        if np.dot(basis[i],new_basis)==0: 
            if i+1==basis.shape[0]:
                basis=np.vstack((basis,new_basis)) 
        else: 
            break
    return basis

def fix_seed_NMF(seed): 
    np.random.seed(seed)

if __name__=='__main__':

    input_file = "sample_data_for_SONMF.txt"

    basis = 6
    W_dict = {}  # dictionary to store W for each SEED

    for SEED in range(0,100):
        fix_seed_ONMF(SEED)
        bs = 2
        m  = 3
        k  = 2
        A = torch.randn(bs, m, k)
        sgns = binarize(torch.Tensor([i for i in range(2**k)]).int(), k)

        data_load = np.loadtxt(input_file)
        # Convert a NumPy array to a PyTorch tensor
        data_t = torch.tensor(data_load, dtype=torch.float32)
        data = data_t.T
        data_numpy = data.to('cpu').detach().numpy().copy()

        W, H = onmf(data, basis, rank=10, max_iter=1000)
        WH = W @ H
        w_numpy = W.to('cpu').detach().numpy().copy()
        h_numpy = H.to('cpu').detach().numpy().copy()
        wh_numpy = WH.to('cpu').detach().numpy().copy()

        # Binarize using a threshold
        threshold = 0.1
        W_dict[SEED] = np.where(w_numpy >= threshold,1,0)


    w_all = None

    for SEED in range(0,100):
        if SEED == 0:
            # Process 1st SEED (set first w_0)
            w_0 = W_dict[SEED]
            w_0_t = w_0.T  
            w_all = w_0_t  
        else:
            w = W_dict[SEED]  
            w_t = w.T  
            w_all = np.append(w_all, w_t, axis=0)


    count_list = np.empty((0, 2), dtype=object)  # array (elements and number of occurrences)

    while len(w_all) > 0:
        index = [0]
        criterion = w_all[0]
        
        for i in range(1, len(w_all)):
            if all(w_all[i] == criterion):
                index.append(i)
        
        count_index = len(index)
        
        # Store only if count_index is non-zero
        if count_index != 0:
            count_list = np.append(count_list, np.array([[criterion, count_index]]), axis=0)
        
        # Remove elements once stored from w_all
        for idx in sorted(index, reverse=True):
            w_all = np.delete(w_all, idx, 0)

    # Sort in descending order according to count_index
    sorted_list = count_list[count_list[:, 1].argsort()[::-1]]

    w_candi = sorted_list[:, 0]  

    # Create basis matrix
    w = np.empty((0, len(w_candi[0])), dtype=np.float64)  
    for i in range(w_candi.shape[0]):
        if not all(w_candi[i]==0):
            if len(w)==0:
                w=w_candi[i][np.newaxis,:].astype(np.float64)  
            elif w.shape[0]<basis:
                w=orthogonal_check(w,w_candi[i].astype(np.float64))  
            else:
                break

    W = np.ascontiguousarray(w.T)  # orthogonal basis matrix (WIN)  
    if W.shape[1]<basis:
        print(f'Only {W.shape[1]} bases were obtained.')
    else:
        np.save('WIN.npy',W)

    # NMF
    nmf = NMF(n_components=basis , init = 'custom')

    SEED = 0
    fix_seed_NMF(SEED)

    # initial basis matrix (WIN)
    w_init = W 

    # random initial coefficient matrix
    h_init = np.random.rand(basis,data.shape[1])

    W = nmf.fit_transform(data, W = w_init , H = h_init)
    H = nmf.components_
    WH = W @ H

    # Results output
    np.save("W", W)
    np.save("H", H)
    np.save("WH", WH)

    # MSE evaluation
    mse = mean_squared_error(data, WH) 
    print("mse",mse)