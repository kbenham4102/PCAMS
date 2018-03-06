import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from scipy import sparse
from scipy.sparse import linalg
import scipy.optimize as optimize
#from peakutils import baseline
# for n x p data with n features and p repetitions. For mass spectrometry data,
# each row contains intensites for each mass-to-charge, each column contains the
# intensities for a spectrum. (e.g. n x p, n masses, p spectra)
# Ex. syntax => X = pFeaturesnReps(DataMatrix)
def properties(DataMatrix):
    # Determine mean of each row's values
    mean = np.mean(DataMatrix, axis = 1)
    # Determine standard deviation of each column's values
    stddevC = np.std(DataMatrix, axis = 0, ddof = 1)
    # Determine standard deviation of row's values
    stddevR = np.std(DataMatrix, axis = 1, ddof = 1)
    minvalstd = np.min(stddevR[np.nonzero(stddevR)])
    for i in range(0, len(stddevR)):
        if stddevR[i] == 0:
            stddevR[i] = 1
    return mean, stddevR, stddevC

# Returns a difference matrix for 2 replicates arranged in adjacent columns
# Cols 1,2 => set 1 of replicates, Cols 3,4 = set 2 of reps.....
def diffmat2(DataMatrix):
    n,p = DataMatrix.shape
    p2 = int(p/2)
    D = np.zeros((n,p2))
    j = 0
    for i in range(0,p,2):
        D[:,j] = abs(DataMatrix[:,i] - DataMatrix[:,i+1])
        j += 1
    return D
# Specific averaging function for a certain replicate data set.
def avgmat(DataMatrix, numreps):
    n,p = DataMatrix.shape
    p2 = int(p/numreps)
    A = np.zeros((n,p2))
    j = 0
    for i in range(0,p,numreps):
        A[:,j] = np.mean(DataMatrix[:,:(i+3)], axis = 1)
        j += 1
    return A

# Returns centered data
def center(DataMatrix):
    n,p = DataMatrix.shape
    m = np.mean(DataMatrix, axis = 1, keepdims = True)
    DataMatrix -= m
    return DataMatrix

# Returns centered and standardized by row deviation data
def stdizerow(DataMatrix):
    n,p = DataMatrix.shape
    DataMatrix = center(DataMatrix)
    sig = np.std(DataMatrix, axis = 1, ddof = 1, keepdims = True)
    DataMatrix /= sig
    return DataMatrix

# Returns centered and standardized by column deviation data
def stdizecol(DataMatrix):
    n,p = DataMatrix.shape
    sig = np.std(DataMatrix, axis = 0, ddof = 1, keepdims = True)
    DataMatrix = center(DataMatrix)
    DataMatrix /= sig
    return DataMatrix

# Centers and returns covariance matrix, only uses X.T * X so in order to
# obtain desired covariance dimension input must be transposed in some cases.
def covariance(DataMatrix):
    n,p = DataMatrix.shape
    X = np.mat(center(DataMatrix))
    covmatrix = X.T*X/(n-1)
    return covmatrix

# Normalizes columns (individual spectra) of DataMatrix by highest value within
# a column (spectra). Data must have intensities for each m/z arranged as cols.
def normspectra(DataMatrix):
    n,p = DataMatrix.shape
    maximums = np.amax(DataMatrix, axis = 0, keepdims = True)
    DataMatrix /= maximums
    return DataMatrix
# Normalizes each sample by the sum of all features
def totionnorm(DataMatrix):
    n,p = DataMatrix.shape
    totcurrents = np.sum(DataMatrix, axis = 0, keepdims = True)
    DataMatrix /= totcurrents
    return DataMatrix

# Normalizes each sample by first squaring each element and dividing by the
# sum of squared features
def ionsqnorm(DataMatrix):
    n,p = DataMatrix.shape
    currentsq = np.sum(DataMatrix**2, axis = 0, keepdims = True)
    DataMatrix /= currentsq
    return DataMatrix


# Probabilistic Quotient Normalization
# See Dieterle et al. Analytical Chemistry, Vol. 78, No. 13, July 1, 2006
# Function requires a set of reference spectra in ControlMat, and spectra
# to be normalized in DataMatrix.
def pqnorm(ControlMat, DataMatrix):
    n,p = DataMatrix.shape
    DataMatrix = 100*totionnorm(DataMatrix)
    ControlMat = 100*totionnorm(ControlMat)
    control = np.median(ControlMat, axis = 1)
    DataNorm = np.empty((n,p))
    meds = np.empty(p)
    for i in range(p):
        meds[i] = np.median(DataMatrix[:,i]/control)
        DataNorm[:,i] = DataMatrix[:,i]/meds[i]
    return DataNorm



# This is a baseline correction function based on the asymmetric least squares
# baseline correction theory. Paper citation Eilers, Boelens, "Baseline
# Correction with Asymmetric Least Squares Smoothing". 2005. Google has the
# paper for free.
# DataMatrix - n x p array
# lam - lamda parameter, values range from 10^2 - 10^9 typically
# p - penalty parameter, values range from 1e-4 to 0.1 typically
# tol - convergence tolerance
# bslines - if true, returns DataFixed and baselines arrays. Default is False.
def baselinecorr(DataMatrix, lam, p, tol, bslines = False):
    m,n = DataMatrix.shape
    DataFixed = np.empty((m,n))
    baselines = np.empty((m,n))
    D = sparse.csc_matrix(np.diff(np.eye(m), 2))
    D1 = D
    w1 = np.ones(m)
    print("Starting iterative baseline determination")
    for l in range(len(DataMatrix[0,:])):
        y = DataMatrix[:,l]
        k = 0
        i = 1
        D = D1
        while i > tol:
            W = sparse.spdiags(w1, 0, m, m)
            Z = W + (lam * sparse.csc_matrix.dot(D, D.T))
            z = linalg.spsolve(Z, w1*y)
            w2 = p * (y > z) + (1-p) * (y < z)
            i = np.sum((w2 - w1)**2)
            w1 = w2
            k += 1
        DataFixed[:,l] = y - z
        baselines[:,l] = z
        #print("%d iterations" % k)
    if bslines == True:
        return DataFixed, baselines
    else:
        return DataFixed

def baseline(mzs, spectra, deg = 5):
    X = np.ones((len(spectra), deg + 1))
    for i in range(1, deg):
        X[:,i] = mzs**i
    res = np.linalg.lstsq(X, spectra)
    theta = res[0].reshape(deg+1, 1)
    bsline = np.dot(X, theta)
    return bsline



# Polynomial regression to determine baseline, for 20,000 point MALDI data,
# order of k = 5 seems to be appropriate.
def bslineofset(DataMatrix, k):
    m,n = DataMatrix.shape
    DataCorr = np.empty((m,n))
    for i in range(n):
        DataCorr[:,i] = DataMatrix[:,i] - baseline(DataMatrix[:,i], deg = k)
    return  DataCorr


# for n x p data as described above. If needed, centering or standardizing must
# be done beforehand.
class genpcs(object):
    def __init__(self, InputData):
        self.InputData = InputData
        self.dim = InputData.shape
        n,p = self.dim

        U, s, Vt = LA.svd(InputData, full_matrices = False)
        V = Vt.T
        S = np.mat(np.diag(s))
        self.singvals = s
        self.svalvar = s/np.sum(s)
        self.evalsSVD = s**2/(n-1)
        self.S = S
        self.V_SVD = V
        self.Vt = Vt
        self.U = U
        self.co_svd = np.dot(np.dot(np.dot(V,S),U.T), np.dot(
                                            np.dot(U,S), Vt))/(n-1)
        self.PCf = np.dot(U,S)
        self.PCs = np.dot(S,Vt).T
    # Reconstruct original data using k principal components
    def reconstruct(self, k):
        U = self.U
        S = self.S
        Vt = self.Vt
        reconst = U[:,:k]*S[:k,:k]*Vt[:k,:]
        return reconst

    def desc_variance(self):
        varnum = 0
        coverage = 0
        s = self.singvals
        while coverage < 0.95:
            coverage += s[varnum]/np.sum(s)
            varnum += 1
        string = ("%d components describe %f percent of the variance"
                                    % (varnum, coverage*100))
        return string


# Determine outliers from 2D score plot of the features
    def outliersPC1PC2(self, opt = False):
        def geomcenter(X, Y):
            def geommedian(P, X, Y):
                n = len(X)
                d = np.zeros(n)
                for i in range(n):
                    d[i] = np.sqrt((P[0] - X[i])**2 + (P[1] - Y[i])**2)
                return np.sum(d)
            gm = optimize.minimize(geommedian, [1, 1], args = (X, Y))
            return gm.x

        def dfrompoint(X, Y, point):
            dists = np.empty(len(X))
            for i in range(len(X)):
                dists[i] = np.sqrt((point[0] - X[i])**2 + (point[1] - Y[i])**2)
            return dists
        PC1 = np.ravel(self.PCf[:,0])
        PC2 = np.ravel(self.PCf[:,1])
        if opt == True:
            c0 = geomcenter(PC1, PC2)
        else:
            c0 = np.array([0,0])
        dist = dfrompoint(PC1, PC2, c0)
        mean = np.mean(dist)
        sig = np.std(dist)
        k = 0
        tol = 10
        for i in range(0,len(dist)):
            if dist[i] > (tol*sig) + mean:
                    k += 1
        print('''There are %d points over %d standard deviations from the mean
in the score plot.''' % (k, tol))
        outlier_indices = np.argsort(dist)[::-1]
        return outlier_indices

# Determine outliers from 3D score plot of the features
    def outliersPC1PC2PC3(self, opt = False):
        def geomcenter3D(X, Y, Z):
            def geommedian(P, X, Y, Z):
                n = len(X)
                d = np.zeros(n)
                for i in range(n):
                    d[i] = np.sqrt((P[0] - X[i])**2 + (P[1] - Y[i])**2 + (P[2] - Z[i])**2)
                return np.sum(d)
            gm = optimize.minimize(geommedian, [1, 1, 1], args = (X, Y, Z))
            return gm.x


        def dfrompoint(X, Y, Z, point):
            dists = np.empty(len(X))
            for i in range(len(X)):
                dists[i] = np.sqrt((point[0] - X[i])**2 + (point[1] - Y[i])**2
                + (point[2] - Z[i])**2)
            return dists
        PC1 = np.ravel(self.PCf[:,0])
        PC2 = np.ravel(self.PCf[:,1])
        PC3 = np.ravel(self.PCf[:,2])
        if opt == True:
            c0 = geomcenter3D(PC1, PC2, PC3)
        else:
            c0 = np.array([0,0,0])
        dist = dfrompoint(PC1, PC2, PC3, c0)
        mean = np.mean(dist)
        sig = np.std(dist)
        k = 0
        tol = 10
        for i in range(0,len(dist)):
            if dist[i] > (tol*sig) + mean:
                    k += 1
        print('''There are %d points over %d standard deviations from the mean
in the score plot.''' % (k, tol))
        outlier_indices = np.argsort(dist)[::-1]
        return outlier_indices


    def plotscree(self):
        evals = self.evalsSVD
        evals = evals/np.sum(evals)
        number = np.arange(len(evals))
        fig = plt.figure(num = 1)
        ax1 = fig.add_subplot(111)
        ax1.scatter(number, evals, s = 2, c = 'b')
        ax1.set_xlabel('Eigenvalue Index')
        ax1.set_ylabel('Percent of Variance')
        ax1.set_title('Scree Plot')
        plt.show()

# This set of functions is for plotting the features' score plots
    def plotPC1PC2(self):
        PC1 = self.PCf[:,0]
        PC2 = self.PCf[:,1]
        fig = plt.figure(num = 1)
        ax1 = fig.add_subplot(111)
        ax1.scatter(PC1, PC2, s = 2, c = 'b', label = 'PC1,PC2')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_title('PC1 vs PC2')
        plt.legend(loc = 'best')
        plt.show()

    def plotPC1PC3(self):
        PC1 = self.PCf[:,0]
        PC3 = self.PCf[:,2]
        fig = plt.figure(num = 1)
        ax1 = fig.add_subplot(111)
        ax1.scatter(PC1, PC3, s = 2, c = 'b', label = 'PC1,PC3')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC3')
        ax1.set_title('PC1 vs PC3')
        plt.legend(loc = 'best')
        plt.show()

    def plotPC2PC3(self):
        PC2 = self.PCf[:,1]
        PC3 = self.PCf[:,2]
        fig = plt.figure(num = 1)
        ax1 = fig.add_subplot(111)
        ax1.scatter(PC2, PC3, s = 2, c = 'b', label = 'PC2,PC3')
        ax1.set_xlabel('PC2')
        ax1.set_ylabel('PC3')
        ax1.set_title('PC2 vs PC3')
        plt.legend(loc = 'best')
        plt.show()

    def plotPC3PC4(self):
        PC3 = self.PCf[:,2]
        PC4 = self.PCf[:,3]
        fig = plt.figure(num = 1)
        ax1 = fig.add_subplot(111)
        ax1.scatter(PC3, PC4, s = 2, c = 'b', label = 'PC3,PC4')
        ax1.set_xlabel('PC3')
        ax1.set_ylabel('PC4')
        ax1.set_title('PC3 vs PC4')
        plt.legend(loc = 'best')
        plt.show()

# To compare a 'numpop' population spectra set where numspec1 is the number of
# columns of spectra belonging to each population, must be equal. Default principle 
# components to plot are 1 and 2.
    def PCplotlabeled(self, numspec1, numpop, Xpc = 1, Ypc = 2,
    title = 'Score Plot'):
            PCs = np.asarray(self.PCs)
            Xpc = Xpc - 1 #Correct to zero indexing
            Ypc = Ypc - 1
            X = np.zeros((numspec1, numpop))
            Y = np.zeros((numspec1, numpop))

            for i in range(numpop):
                X[:,i] = PCs[i*numspec1:(i+1)*numspec1, Xpc]
                Y[:,i] = PCs[i*numspec1:(i+1)*numspec1, Ypc]

            fig = plt.figure(num = 1)
            ax1 = fig.add_subplot(111)
            for i in range(numpop):
                ax1.scatter(X[:,i], Y[:,i], s = 2, label = "P" + str(i+1))
            ax1.set_xlabel('PC%d' % (Xpc + 1))
            ax1.set_ylabel('PC%d' % (Ypc + 1))
            ax1.set_title(title)
            plt.legend(loc = 'best')
            plt.show()

# Plot the unlabeled score plot of the samples
    def plotPC1PC2sam(self):
        PC1 = self.PCs[:,0]
        PC2 = self.PCs[:,1]
        fig = plt.figure(num = 1)
        ax1 = fig.add_subplot(111)
        ax1.scatter(PC1, PC2, s = 2, c = 'b', label = 'PC1,PC2')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_title('PC1 vs PC2')
        plt.legend(loc = 'best')
        plt.show()
