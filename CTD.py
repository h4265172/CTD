import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D
import math
from sklearn.mixture import GaussianMixture

import data_loader

class CTD():
    def __init__(self,
                 Z_obs,
                 M,
                 AR_lags,
                 difference,
                 K,
                 seed=None,
                 lambda_all=0.0001,
                 lambda_x=0.001,
                 lambda_y=0.001,
                 rbsize=1000,
                 cbsize=300,
                 lr=0.003,
                 iter_per_message=50,
                 pre_patience=5,
                 patience=5,
                 device="cuda:0"
                ):
        
            
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device=device
            
        self.Z_obs = Z_obs
        self.mask = (~torch.isnan(Z_obs)).to(int).to(device)
        self.N, self.T = Z_obs.shape
        self.seed = seed  
        self.data = DataLoader(Z_obs,
                               device=self.device,
                               seed=self.seed,
                               rbsize=rbsize,
                               cbsize=cbsize,
                               shuffle=True)
        
        if not torch.is_tensor(AR_lags):
            AR_lags = torch.tensor(AR_lags)
        self.AR_lags = AR_lags.sort()[0].to(int).to(self.device)
        
        self.difference = difference
        self.M = M
        self.K = K
        self.lambda_all = lambda_all
        self.lambda_x = lambda_x
        self.lambda_y = lambda_y     
        self.lr = lr  
        self.iter_per_message=iter_per_message
        self.pre_patience = pre_patience
        self.patience = patience
        self.pre_patience_count = pre_patience
        self.patience_count = patience
        
        #placeholders for uninitialized attributes
        
        #X and Y
        torch.manual_seed(self.seed)
        self.X = torch.randn((self.N, self.M), device=self.device)
        #self.X.requires_grad_()
        self.Y = torch.randn((self.T, self.M), device=self.device)
        #self.Y.requires_grad_()
        
        #Cluster-related
        self.centers = torch.empty((self.K, self.M), device=self.device)
        self.covs = torch.empty((self.K, self.M, self.M), device = self.device)
        self.props = torch.ones(self.K, device=self.device) / self.K
        self.soft_labels = torch.zeros((self.N, self.K),device=self.device)
        self.current_gmm_model = None
        #temporal
        self.Theta = torch.zeros((self.AR_lags.shape[0] * self.M + 1, 
                                  self.M), device=self.device)
        self.current_gmm_model = None
        #record keeping
        self.pretrain_loop_count = 0
        self.main_loop_count = 0
        self.pretrain_losses = []
        self.cluster_regs = []
        self.temporal_regs = []
        self.overall_losses = [torch.inf]
        
    
    def fit(self, noise=0.2, max_loop=500):
        torch.manual_seed(self.seed)
        self.pretrain(batch=False)
        
        #add noise to pretrained X and Y
        self.X += noise * torch.randn_like(self.X, device=self.device)
        self.Y += noise * torch.randn_like(self.Y, device=self.device)
        
        while (self.patience_count > 0) and (self.main_loop_count < max_loop):
            self.main_loop_count += 1
            print(self.main_loop_count, end="\r")
            self.data.next_batch()
            self.centers, self.covs, self.props = self._cluster_params()
            self._step_DV()
            self._soft_labels()
            self._step_XY()
            self.Theta = self._temporal_params()
            
            #message
            if self.main_loop_count % self.iter_per_message == 0:
                cluster_reg = round(self.cluster_regs[-1], 4)
                temporal_reg = round(self.temporal_regs[-1], 4)
                loss = round(self.overall_losses[-1], 4)

                print("Loop {} | overall_loss {} | x {} | y {} ".format(
                                                 self.main_loop_count, loss, cluster_reg, temporal_reg))
                '''
                plt.clf()
                X_np = self.X.detach().cpu().numpy()
                new_centers = self.centers.detach().cpu().numpy()
                hard_membership = self.soft_labels.argmax(axis=1).detach().cpu().numpy()
                fig, axes = plt.subplots(M-1, M-1, figsize=(4,4),
                                        sharex=True, sharey=True)
                for m1 in range(M):
                    for m2 in range(M):
                        if m1 < m2:
                            axes[m1,m2-1].scatter(X_np[:,m1],X_np[:,m2], 
                                                  c=hard_membership, alpha=0.5)
                            axes[m1,m2-1].scatter(new_centers[:,m1],new_centers[:,m2], 
                                          color="red", marker="+")
                fig.suptitle("Original X")
                plt.show()
                '''
                print("_________________________")
            
            #stopping criterion
            if self.overall_losses[-2] <= self.overall_losses[-1] + 1e-18:
                self.patience_count -= 1
            else:
                self.patience_count = self.patience
            
        print("Training finished")
            
            
    #######################################
    #R functions
    #######################################

    def _angles(self, raw_angles=None):
        raw_angles = self.raw_angles if raw_angles is None else raw_angles
        return math.pi * F.tanh(raw_angles) / 4.
    
    def _two_dim_rotation_mat(self, angle):
        output = torch.empty(2, 2, device=self.device)
        output[0, 0] = torch.cos(angle)
        output[0, 1] = -torch.sin(angle)
        output[1, 0] = torch.sin(angle)
        output[1, 1] = torch.cos(angle)
        return output
    
    def _rotation_mat(self, VTnew_flat=None):
        VTnew_flat = self.VTnew_flat if VTnew_flat is None else VTnew_flat
        angles = math.pi * F.tanh(VTnew_flat) / 4.
        base = torch.eye(self.M, device=self.device)
        for m in range(self.M - 1):
            tmp = torch.eye(self.M, device=self.device)
            tmp[m:(m+2),m:(m+2)] = self._two_dim_rotation_mat(angle=angles[m])
            base = base.matmul(tmp)
        return base
    
    def _center_dist_loss(self, VT_flat):
        VT_mat = self._rotation_mat(VT_flat)
        rotated_centers = self.centers.mm(VT_mat)
        centers_dists = torch.cdist(rotated_centers, rotated_centers)
        return -torch.linalg.norm(centers_dists)
    
    def _dim_weights(self):
        with torch.no_grad():
            overall_center = torch.zeros(self.M, device=self.device)
            for k in range(self.K):
                overall_center += self.props[k] * self.centers[k,:]
            overall_cov = self.covs
            centered_centers = self.centers - overall_center
            centers_cov = centered_centers.T.mm(self.props.diag()).mm(centered_centers)
            weights = torch.empty(self.M, device=self.device)
            
            Imat = torch.eye(M, device=self.device)
            for m in range(self.M):
                vec = Imat[m, :].reshape(1,-1)
                num = vec.mm(centers_cov).mm(vec.T)
                denom = vec.mm(overall_cov).mm(vec.T)
                weights[m] = num/denom          
        return weights
    
    def _step_DV(self, patience=1, loop=40):
        weights = self._dim_weights()
        scale = ((1/ weights.norm()) * weights).float()
        rotation_angles = torch.rand(M-1, device=self.device).float()
        rotation_angles.requires_grad_()
        optim_rotate = optim.Adam([rotation_angles], lr=1.)
        
        for _ in range(loop):
            optim_rotate.zero_grad()
            rotation_mat = self._rotation_mat(rotation_angles)
            rotated_centers = self.centers.mm(rotation_mat)
            centers_dists = torch.cdist(rotated_centers, rotated_centers)
            loss = - torch.linalg.norm(centers_dists)

            loss.backward()
            optim_rotate.step()
        
        angles_eval = rotation_angles.clone().detach()
        rotation_mat_eval = self._rotation_mat(angles_eval)
        R = scale.diag().mm(rotation_mat_eval)
        R_inv = rotation_mat_eval.T.mm((1/scale).diag())
        XR = self.X.clone().detach().mm(R)
        centers_tmp, covs_tmp, self.props = self._cluster_params(XR)
        self._soft_labels()
        
        '''
        plt.clf()
        XR_np = XR.detach().cpu().numpy()
        new_centers_R = centers_tmp.detach().cpu().numpy()
        hard_membership = self.soft_labels.argmax(axis=1).detach().cpu().numpy()
        fig, axes = plt.subplots(M-1, M-1, figsize=(4,4),
                                sharex=True, sharey=True)
        for m1 in range(M):
            for m2 in range(M):
                if m1 < m2:
                    axes[m1,m2-1].scatter(XR_np[:,m1],XR_np[:,m2], 
                                          c=hard_membership, alpha=0.5)
                    axes[m1,m2-1].scatter(new_centers_R[:,m1],new_centers_R[:,m2], 
                                  color="red", marker="+")
        fig.suptitle("XR")
        plt.show()
        '''
        
        self.centers = centers_tmp.mm(R_inv)
        self.covs = R_inv.T.mm(covs_tmp).mm(R_inv)

    #######################################
    #Cluster functions
    #######################################
    
    def _cluster_params(self, X=None, batch=True):
        with torch.no_grad():
            if X is None:
                if batch:
                    X = self.X[self.data.row_idx, :].clone().detach().cpu().numpy()
                else:
                    X = self.X.clone().detach().cpu().numpy()
            else:
                X = X.detach().cpu().numpy()

            gmm = GaussianMixture(n_components=self.K, 
                                  covariance_type="tied", 
                                  random_state=self.seed).fit(X)
            centers = torch.tensor(gmm.means_, device=self.device).float()
            covs = torch.tensor(gmm.covariances_, device=self.device).float()
            props = torch.tensor(gmm.weights_, device=self.device).float()
        
        self.current_gmm_model = gmm
        return centers, covs, props
    
    def _soft_labels(self):
        with torch.no_grad():
            log_props = self.props.log()

            kernels = torch.empty((self.N, self.K), device=self.device)
            prec_logdets = torch.empty(self.K, device=self.device)
            for k in range(self.K):
                #prec_k = torch.linalg.inv(self.covs[k,:,:])
                prec_k = torch.linalg.inv(self.covs)
                prec_logdets[k] = prec_k.logdet()
                centered_X = self.X - self.centers[k,:]
                kernels[:,k] = centered_X.mm(prec_k).mm(centered_X.T).diag()
            
            log_ratios = torch.empty((self.N, self.K, self.K), device=self.device)
            for k1 in range(self.K):
                log_ratios[:, k1, k1] = 0.
                for k2 in range(self.K):
                    if k1 < k2:
                        log_prop_diff = log_props[k2] - log_props[k1]
                        log_likelihood_diff = -0.5 * ((kernels[:,k2] - kernels[:, k1]) + \
                                             (prec_logdets[k2] - prec_logdets[k1]))
                        log_ratios[:, k1, k2] = log_prop_diff + log_likelihood_diff
                        log_ratios[:, k2, k1] = -(log_prop_diff + log_likelihood_diff)

            self.soft_labels = 1 / (torch.nan_to_num(log_ratios.exp(), nan=0.).sum(dim=2))
        
    def cluster_loss_fun(self, X, return_grad=True):
        prec_dets_sqrt = torch.empty(self.K, device=self.device)
        tmp = torch.empty((X.shape[0], self.M, self.K), device=self.device)
        kernels = torch.empty((X.shape[0], self.K), device=self.device)
        for k in range(self.K):
            #prec_k = torch.linalg.inv(self.covs[k,:,:])
            prec_k = torch.linalg.inv(self.covs)
            prec_dets_sqrt[k] = prec_k.det().sqrt()
            centered_X = X - self.centers[k,:]
            tmp[:, :, k] = centered_X.mm(prec_k)
            kernels[:, k] = tmp[:, :, k].mm(centered_X.T).diag()
        
        loss = -self.lambda_x * (((-kernels).exp() * \
                                 (self.props * prec_dets_sqrt)).sum(dim=1)+1e-8).log().mean()
        
        grad = self.lambda_x * (tmp * self.soft_labels[self.data.row_idx, :].reshape(-1, 1, self.K)).sum(dim=2)
        
        if return_grad:
            return loss, grad
        else:
            return loss, None
    
    #######################################
    #temporal functions
    #######################################
    
    def _temporal_params(self, Y=None, batch=True):
        if Y is None:
            if batch:
                Y = self.Y[self.data.col_idx[0]:self.data.col_idx[1], :].clone().detach()
            else:
                Y = self.Y.clone().detach()
        else:
            Y = Y
        
        with torch.no_grad():
            for _ in range(self.difference):
                Y = Y.diff(dim=0)
            T, _ = Y.shape
            past = torch.empty((T - self.AR_lags.max(), 
                                (self.AR_lags.shape[0] * Y.shape[1] + 1)), device="cuda:0").float()
            past[:,0] = 1.
            for i, l in enumerate(self.AR_lags):
                past[:, (i)*M + 1: (i+1)*M + 1] = Y[(self.AR_lags.max() - l): -l]
            inv = torch.linalg.inv(past.T.mm(past)).float()
            sol = inv.mm(past.T).mm(Y[self.AR_lags.max():])
        return sol
    
    def temporal_loss_fun(self, Y):
        for _ in range(self.difference):
            Y = Y.diff(dim=0)
        T, _ = Y.shape
        past = torch.empty((T - self.AR_lags.max(), 
                           (self.AR_lags.shape[0] * self.M + 1)), 
                                device="cuda:0").to(float)
        past[:,0] = 1.
        for i, l in enumerate(self.AR_lags):
            past[:, (i * self.M + 1): ((i + 1) * self.M + 1)] = Y[(self.AR_lags.max() - l): -l]
        Y_hat = past.float().mm(self.Theta.float())
        
        return (self.lambda_y / (T - self.AR_lags.max())) * torch.norm(Y[self.AR_lags.max():]-Y_hat).pow(2)
    
    #######################################
    #X and Y functions
    #######################################
    
    def main_loss_fun(self, X, Y, obs, mask):
        return (1 / mask.sum()) * torch.norm(mask * (X.mm(Y.T) - torch.nan_to_num(obs, 0))).pow(2)
        
    def _step_XY(self, batch=True):
        torch.autograd.set_detect_anomaly(True)
        
        if batch:
            X = self.X[self.data.row_idx, :].clone().detach()
            Y = self.Y[self.data.col_idx[0]:self.data.col_idx[1], :].clone().detach()
            data = self.data.batch
            mask = self.data.batch_mask
        else:
            X = self.X.clone().detach()
            Y = self.Y.clone().detach()
            data = self.Z_obs
            mask = self.data.mask

        X.requires_grad_()
        Y.requires_grad_()
        optim_XY = optim.Adam([X, Y], lr=self.lr)
        optim_XY.zero_grad()
        
        main = self.main_loss_fun(X, Y, data, mask)
        cluster_reg, cluster_grad = self.cluster_loss_fun(X)
        temporal_reg = self.temporal_loss_fun(Y)
        
        loss = temporal_reg + main + self.lambda_all * (Y.norm()/self.N + X.norm()/self.T) #+ cluster_reg
        loss.backward()
        
        #Manually compute cluster_reg gradient to avoid overflow
        #tmp = torch.empty(self.N)
        
        X.grad += cluster_grad.clone().detach()
        optim_XY.step()
        
        if torch.isinf(cluster_reg):
            print("Loop {}: There is nan from the recent update".format(self.main_loop_count))
            print(kernels[kernels.mean(dim=1).argmax(),:])
            
        self.overall_losses.append(loss.item()+cluster_reg.item())
        self.cluster_regs.append(cluster_reg.item())
        self.temporal_regs.append(temporal_reg.item())
        
        if batch:
            self.X[self.data.row_idx, :] = X.clone().detach()
            self.Y[self.data.col_idx[0]:self.data.col_idx[1], :] = Y.clone().detach()
        else:
            self.X = X.clone().detach()
            self.Y = Y.clone().detach()
        
    #######################################
    #pretraining functions
    #######################################
    def pretrain(self, batch=True):
        current_loss = torch.inf
        
        print("##################################")
        print("Pretraining begins")
        print("##################################")
        
        while self.pre_patience_count > 0:
            self.pretrain_loop_count += 1
            if batch:
                self.data.next_batch()
                X = self.X[self.data.row_idx, :].clone().detach()
                Y = self.Y[self.data.col_idx[0]:self.data.col_idx[1], :].clone().detach()
                data = self.data.batch
                mask = self.data.batch_mask
            else:
                X = self.X.clone().detach()
                Y = self.Y.clone().detach()
                data = self.Z_obs
                mask = self.data.mask
            X.requires_grad_()
            Y.requires_grad_()
            pre_optim_XY = optim.Adam([X, Y], lr=self.lr)
            pre_optim_XY.zero_grad()
            loss = self.main_loss_fun(X, Y, data, mask)
            #print(loss)
            if loss < current_loss:
                loss.backward()
                pre_optim_XY.step()
                current_loss = loss.detach().item()
                self.pretrain_losses.append(current_loss)
                if batch:
                    self.X[self.data.row_idx, :] = X.clone().detach()
                    self.Y[self.data.col_idx[0]:self.data.col_idx[1], :] = Y.clone().detach()
                else:
                    self.X = X.clone().detach()
                    self.Y = Y.clone().detach()
                
                self.pre_patience_count = self.pre_patience
            
            else:
                self.pre_patience_count -= 1
            
            #print("Pretrain loop: {}, patience: {}".format(self.pretrain_loop_count,
            #                                               self.pre_patience_count))
            if self.pretrain_loop_count % 2000 == 0:
                print("Loop {} | pretrain loss {}".format(self.pretrain_loop_count, 
                                                          round(self.pretrain_losses[-1], 4)), end='\r')
                
        print("Loop {} | pretrain loss {}".format(self.pretrain_loop_count, loss))
        self.X.requires_grad = False
        self.X.requires_grad = False
        self.Y.requires_grad = False
        self.XU, self.Xsing_vals, self.XVT = torch.linalg.svd(self.X, full_matrices=False)
        
        #self.Xkappa = self.Xsing_vals.max()/self.Xsing_vals.min()
        self.XU.requires_grad = False
        self.Xsing_vals.requires_grad = False
        self.XVT.requires_grad = False
        print("X singular values: {}". format(self.Xsing_vals))
        #print("X Condition #: {}".format(self.Xkappa))
        print("#####################################")
        print("Pretraining is finished")
        print("#####################################")
    
    
    #######################################
    #after training
    #######################################
    def Z_hat(self):
        return self.X.mm(self.Y.T)
    
    def predict_Y(self, tau):
        lookback = self.AR_lags.max() + self.difference
        diff_Y = self.Y.clone().detach()[-lookback:]
        
        for _ in range(self.difference):
            diff_Y[1:] = diff_Y.diff(dim=0)
        diff_Y = torch.vstack((diff_Y, torch.nan * torch.empty((tau, self.M),
                                                              device=self.device)))
        
        past = torch.empty(tau, (self.AR_lags.shape[0] * self.M + 1), device=self.device)
        past[:, 0] = 1.
        
        for t in range(tau):
            idx_for_t = lookback+t-self.AR_lags
            past[t, 1:] = diff_Y[idx_for_t].reshape(1, -1)
            diff_Y[lookback + t, :] = (past[t, :].reshape(1, -1)).mm(self.Theta)
        
        Y = diff_Y
        for _ in range(self.difference):
            Y = Y.cumsum(dim=0)
        return Y[-tau:]
            
    def predict(self, tau):
        Y_future = self.predict_Y(tau)
        return self.X.mm(Y_future.T)

    def compute_cluster_error(self, true_labels):
        unscaled_accuracy = 0.
        loop_count=0
        for perm in multiset_permutations(range(self.K)):
            print("computing accuracy: {}%".format(loop_count/math.factorial(self.K)), end='\r')
            tmp = 0.
            for k in range(self.K):
                tmp += torch.sum(self.soft_labels[true_labels == k, perm[k]]).detach().item()
            if tmp > unscaled_accuracy:
                unscaled_accuracy = tmp
            loop_count += 1
        return 1 - unscaled_accuracy / self.N
