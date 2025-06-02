import numpy as np
#import random
import torch
import numpy.random as rnd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.manifold import TSNE
from statsmodels.tsa.arima_process import ArmaProcess

#%%
class Simulate():
    def __init__(self, N, T, Z_sd, p_missing, difference,
                 centers, sigmas, tmp_params=None, 
                 l_max=None, cluster_proportions=None,
                 AR_sd=1., seed=1234, torch_device=None):

        self.device = torch_device

        self.N = N
        self.T = T
        self.difference = difference
        self.seed=seed
        np.random.seed(self.seed)
        

        self.centers = centers
        self.K = centers.shape[0]
        self.M = centers.shape[1]
        self.sigmas = sigmas

        self.tmp_params = tmp_params
        assert self.centers.shape[1] == self.tmp_params.shape[1], "inconsistant M between X and Y; check dim of hyperparameters"


        if cluster_proportions is not None:
            self.cluster_proportions = cluster_proportions
        else:
            num = np.random.exponential(scale=1.0, size=self.K)
            self.cluster_proportions = num/num.sum()
            #adjust to ensure no tiny group
            self.cluster_proportions = (self.cluster_proportions + 1)/(self.cluster_proportions.sum() + self.K)

        self.labels = rnd.choice(np.arange(self.K),self.N, p=self.cluster_proportions)

        self.AR_sd = AR_sd
        self.Z_sd = Z_sd
        self.X = np.empty((self.N, self.M))
        self.Y_AR = np.empty((self.T, self.M))
        self.Y = np.empty((self.T, self.M))
        self.p_missing = p_missing
        self.Z_star = None
        self.mask = None
        self.Z_obs = None
    
    def generate(self):
        np.random.seed(self.seed)
        self._generate_X()
        self._generate_Y()
        self._generate_mask()
        self.Z_star = np.dot(self.X, self.Y.T)
        Z_obs_tmp = self.Z_star + rnd.normal(0, self.Z_sd, size=(self.N, self.T))
        self.Z_obs = np.where(self.mask == 0, np.nan, Z_obs_tmp)
        #print("X,Y,mask, Z, Z_obs have been generated")
        
        if self.device is not None:
            self.centers = torch.from_numpy(self.centers).to(self.device)
            self.sigmas = torch.from_numpy(self.sigmas).to(self.device)
            self.cluster_proportions = torch.from_numpy(
                                        self.cluster_proportions).to(self.device)
            self.tmp_params = torch.from_numpy(self.tmp_params).to(self.device)
            self.X = torch.from_numpy(self.X).to(self.device)
            self.Y = torch.from_numpy(self.Y).to(self.device)
            self.Z_star = torch.from_numpy(self.Z_star).to(self.device)
            if self.mask is not None:
                self.mask = torch.from_numpy(self.mask).to(self.device)
            if self.Z_obs is not None:
                self.Z_obs = torch.from_numpy(self.Z_obs).to(self.device)
            #print("All objects converted to torch tensor")
    
    def _generate_X(self):
        for k in range(self.K):
            self.X[self.labels == k, :] = rnd.multivariate_normal(mean=self.centers[k, :],
                                                                  cov=np.diag(self.sigmas[k,:]**2),
                                                                  size=np.sum(self.labels == k))

    def _generate_Y(self):
        for m in range(self.M):
            AR_generator = ArmaProcess(np.concatenate(([1], -self.tmp_params[1:, m])), np.array([self.AR_sd]))
            assert AR_generator.isstationary, "{}th AR model is not stationary".format(m + 1)

            self.Y_AR[:, m] = AR_generator.generate_sample(nsample=self.T, burnin=100)
            self.Y = self.Y_AR[:,:]
            for _ in range(self.difference):
                self.Y = self.Y.cumsum(axis=0)
            
        self.Y = self.Y/self.Y.std(axis=0)
    
    def _generate_mask(self):
        self.mask = rnd.choice(np.array([0, 1]), replace=True, 
                          size=self.N * self.T, p=[self.p_missing, 1-self.p_missing]).reshape(self.N, self.T)
        
    def plot_X(self):
        if torch.is_tensor(self.X):
            X = self.X.detach().cpu().numpy()
        else:
            X = self.X
        X_df = pd.DataFrame(X)
        if self.M <= 2:
            X_df.columns = ["dim0", "dim1"]
            data = X_df
            title = "Scatter plot of time-invariant embedding"
        else:
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne_result = tsne.fit_transform(X_df)
            data = pd.DataFrame(tsne_result, columns=["dim0", "dim1"])
            title = "t-SNE plot of the time-invariant embedding"

        data['label'] = self.labels
        plt.clf()
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x='dim0', y='dim1',
                        hue='label', palette=sns.color_palette("hls", self.K),
                        data=data, legend="full", alpha=0.3)
        plt.title(title)
        plt.show()
        
    def plot_Y(self, plot_type):
        #possible type "time" and "acf"
        if plot_type == "time":
            plt.clf()
            fig = plt.figure(figsize=(6,4))
            
            for m in range(self.M):
                fig.add_subplot(int("{}1{}".format(self.M+1, m+1)))
                plt.plot(self.Y[:, m])
                if m < self.M-1:
                    plt.xticks(color='w')
            fig.suptitle("Time plot for embedded time series")
            fig.tight_layout(pad=1)
            plt.show()

        if plot_type == "acf":
            plt.clf()
            fig, ax = plt.subplots(self.M, 1, figsize=(6,4))
            fig.suptitle("ACF for embedded time series")
            fig.tight_layout(pad=1.5)
            for m in range(self.M):
                plot_acf(self.Y[:, m], ax=ax[m], title="dim {}".format(m))
                if m < self.M-1:
                    ax[m].set_xticks([])
            plt.show()
    
    def plot_Z(self, Z, plot_type, sample_size=None):
        #possible types: "tsne","time"
        Z_df = pd.DataFrame(Z)

        if plot_type == "tsne":
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne_result = tsne.fit_transform(Z_df)
            data = pd.DataFrame(tsne_result, columns=["dim0", "dim1"])
            data['label'] = self.labels
            plt.clf()
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x='dim0', y='dim1',
                            hue='label', palette=sns.color_palette("hls", self.K),
                            data=data, legend="full", alpha=0.3)
            plt.title("t-SNE plot of rows of Z")
            plt.show()

        else:
            if not sample_size:
                sample_size = 3
            rand_idx = []
            for k in range(self.K):
                rand_idx.append(rnd.choice(np.arange(self.N)[self.labels==k], replace=False))
            colors = sns.color_palette("hls", self.K)

            plt.clf()
            fig = plt.figure(figsize=(8, 5))
            for k in range(self.K):
                member_indices = np.where(self.labels == k)[0]
                random_idx = rnd.choice(member_indices, sample_size, replace=True)
                plt.plot(Z[random_idx, :].T, color=colors[k], alpha=0.5)
                plt.plot([],[],color=colors[k], label ="Cluster {}".format(k))
                plt.legend()
            fig.suptitle("Time plot for sampled Z rows")
            fig.tight_layout(pad=1)
            plt.show()