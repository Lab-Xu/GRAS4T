import numpy as np
from munkres import Munkres
from scipy.sparse.linalg import svds
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, cohen_kappa_score, accuracy_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_kernels


class GCSC_Kernel:

    def __init__(self, A, regu_coef=1., kernel='rbf', gamma=1., ro=0.5, f = 1.5,
                 save_affinity=False, debug_C = False, d = 7, alpha = 18):
        """
        :param regu_coef: regularization coefficient i.e. labmda
        :param n_neighbors: number of neighbors of knn graph
        :param kernel: kernel functions, default 'rbf'. See sklearn for available kernels
        :param gamma: only used in rbf kernel
        :param ro: post-processing parameters
        :param save_affinity: if True, save affinity matrix
        """
        self.regu_coef = regu_coef
        self.kernel = kernel
        self.gamma = gamma
        self.ro = ro
        self.save_affinity = save_affinity
        self.A = A
        self.debug_C = debug_C
        self.f = f
        self.d = d
        self.alpha = alpha

    def __adjacent_mat(self):
        """
        Construct normalized adjacent matrix, N.B. consider only connection of k-nearest graph
        :param x: array like: n_sample * n_feature
        :return:
        """
        # print("Calculating the normalised A matrix...")
        A = self.A * np.transpose(self.A)
        # print("A type: {}, A shape: {}".format(type(A), A.shape))
        D = np.diag(np.reshape(np.sum(A, axis=1) ** -0.5, -1))
        normlized_A = np.dot(np.dot(D, A), D)

        return normlized_A

    def fit(self, X, use_thrC=True):
        # print("Start fitting...")
        A = self.__adjacent_mat()
        if self.kernel == 'linear':
            K = pairwise_kernels(X, metric='linear')
        elif self.kernel == 'polynomial':
            K = pairwise_kernels(X, metric='polynomial', gamma=0.05, degree=3)
        elif self.kernel == 'sigmoid':
            K = pairwise_kernels(X, metric='sigmoid', gamma=0.5)
        elif self.kernel == 'rbf':
            K = pairwise_kernels(X, metric='rbf', gamma=self.gamma) # (N, N)
        else:
            raise Exception('Invalid kernel')
        # print("test, K shape", K.shape)
        I = np.eye(X.shape[0]) # (N, N)
        T = np.dot(np.transpose(A), K) # (N, N)
        inv = np.linalg.inv(np.dot(T, K) + self.regu_coef * I) # (N, N)
        C = np.dot(inv, T)
        
        if use_thrC:
            # print("Modified C matrix...")
            C = self.thrC(C,  self.ro)
        return C

    def thrC(self, C, ro):
        if ro < 1:
            N = C.shape[1]
            Cp = np.zeros((N, N))
            S = np.abs(np.sort(-np.abs(C), axis=0))
            Ind = np.argsort(-np.abs(C), axis=0)
            for i in range(N):
                cL1 = np.sum(S[:, i]).astype(float)
                stop = False
                csum = 0
                t = 0
                while (stop == False):
                    # print(t)
                    csum = csum + S[t, i]
                    # print('csum:{}, ro * cL1:{}'.format(csum, ro * cL1))

                    if csum > ro * cL1:
                        stop = True
                        Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                    t = t + 1
        else:
            Cp = C
        return Cp

    def build_aff(self, C):
        N = C.shape[0]
        Cabs = np.abs(C)
        ind = np.argsort(-Cabs, 0)
        for i in range(N):
            Cabs[:, i] = Cabs[:, i] / (Cabs[ind[0, i], i] + 1e-6)
        Cksym = Cabs + Cabs.T
        return Cksym

    def restoration_C(selfself, C, f):
        debug_matrix = np.ones_like(C)
        dg_C = np.diag(np.diag(C)) / f
        debug_matrix = np.transpose(np.dot(debug_matrix, dg_C))
        debug_C = np.where(C > debug_matrix, C, 0)
        return debug_C

    def post_proC(self, C, K, d, alpha):
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        C = 0.5 * (C + C.T)
        r = d * K + 1
        U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** alpha)
        L = L / L.max()

        if self.debug_C:
            L = self.restoration_C(L, self.f)

        L = 0.5 * (L + L.T)
        return L

