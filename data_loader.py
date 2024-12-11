import torch
import numpy as np

class DataLoader():
    def __init__(
    self,
    Z_obs,
    device,
    seed,
    rbsize=200,
    cbsize=100,
    rend_index=None,
    cend_index=None,
    shuffle=False):
        self.N, self.T, _ = Z_obs.shape
        assert rbsize <= self.N and cbsize <= self.T, \
                'window size ({},{}) must not be larger than data dim ({},{})'.format(rbsize, cbsize, self.N, self.T)
        self.rindex = 0
        self.cindex = 0
        self.epoch = 0
        self.seed = seed
        self.rbsize = rbsize
        self.cbsize = cbsize
        self.mask = (~torch.isnan(Z_obs)).to(int).to(device)
        self.mask.requires_grad = False
        self.Z_obs = torch.nan_to_num(Z_obs).to(device)
        self.Z_obs.requires_grad = False
        
        self.rend_index = self.N if rend_index is None else rend_index
        self.cend_index = self.T if cend_index is None else cend_index
        
        self.shuffle = shuffle
        if self.shuffle:
            np.random.seed(self.seed)
            self.I = np.random.choice(self.N, self.N, replace=False)
        else:
            self.I = np.arange(self.N)
            
        self.batch = "First batch has not been initialized"
        self.batch_mask = "First batch has not been initialized"
        self.row_idx = "First batch has not been initialized"
        self.col_idx = "First batch has not been initialized"
        
    def next_batch(self):
        if self.cindex + self.cbsize >= self.cend_index:
            pr_cindex = self.cindex
            self.cindex = 0
            if self.rindex + self.rbsize >= self.rend_index:
                pr_rindex = self.rindex
                self.rindex = 0
                self.epoch = self.epoch + 1
                if self.shuffle:
                    self.I = np.random.choice(self.N, self.N, replace=False)
            else:
                pr_rindex = self.rindex
                self.rindex = self.rindex + self.rbsize
        else:
            pr_cindex = self.cindex
            self.cindex = self.cindex + self.cbsize
            pr_rindex = self.rindex
        
        
        row_idx = self.I[int(pr_rindex) : int(pr_rindex + self.rbsize)]
        column_end = min(self.cend_index, pr_cindex + self.cbsize)
        col_idx = (int(pr_cindex), int(column_end))
        batch = self.Z_obs[row_idx][:, col_idx[0]:col_idx[1], :]
        batch_mask = self.mask[row_idx][:, col_idx[0]:col_idx[1], :]
        
        self.batch = batch
        self.batch_mask = batch_mask
        self.row_idx = row_idx
        self.col_idx = col_idx
        
        #print("returns data, mask, row_idx, col_idx_range")
        #return data, mask, row_idx, col_idx
