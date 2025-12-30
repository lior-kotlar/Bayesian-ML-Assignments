import numpy as np
from typing import Callable
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

KERNEL_STRS = {
    'Laplacian': r'Laplacian, $\alpha={}$, $\beta={}$',
    'RBF': r'RBF, $\alpha={}$, $\beta={}$',
    'Spectral': r'Spectral, $\alpha={}$, $\beta={}$, $\gamma={}$',
    'NN': r'NN, $\alpha={}$, $\beta={}$'
}


def average_error(pred: np.ndarray, vals: np.ndarray):
    """
    Calculates the average squared error of the given predictions
    :param pred: the predicted values
    :param vals: the true values
    :return: the average squared error between the predictions and the true values
    """
    return np.mean((pred - vals)**2)


def RBF_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the RBF kernel
    :param alpha: the kernel variance
    :param beta: the kernel bandwidth
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        if np.ndim(x) == 1:
            x = x.reshape(-1, 1)
        if np.ndim(y) == 1:
            y = y.reshape(-1, 1)
        
        sqdist = cdist(x, y, metric='sqeuclidean')
        return alpha * np.exp(-beta * sqdist)
    return kern


def Laplacian_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the Laplacian kernel
    :param alpha: the kernel variance
    :param beta: the kernel bandwidth
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        if np.ndim(x) == 1:
            x = x.reshape(-1, 1)
        if np.ndim(y) == 1:
            y = y.reshape(-1, 1)
        dist = cdist(x, y, metric='cityblock')
        return alpha * np.exp(-beta * dist)
    return kern


def Spectral_kernel(alpha: float, beta: float, gamma: float) -> Callable:
    """
    An implementation of the Spectral kernel (see https://arxiv.org/pdf/1302.4245.pdf)
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        if np.ndim(x) == 1:
            x = x.reshape(-1, 1)
        if np.ndim(y) == 1:
            y = y.reshape(-1, 1)
        sqdist = cdist(x, y, metric='sqeuclidean')
        dists = np.sqrt(sqdist)
        return alpha * np.exp(-beta * sqdist) * np.cos(np.pi * dists / gamma)
    return kern


def NN_kernel(alpha: float, beta: float) -> Callable:
    """
    An implementation of the Neural Network kernel (see section 4.2.3 of http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
    :return: a function that receives two inputs and returns the output of the kernel applied to these inputs
    """
    def kern(x, y):
        if np.ndim(x) == 1:
            x = x.reshape(-1, 1)
        if np.ndim(y) == 1:
            y = y.reshape(-1, 1)
        
        dot_products = x @ y.T
        numerator = 2 * beta * (dot_products) + 1
        x_norms_squared = np.sum(x**2, axis=1)
        y_norms_squared = np.sum(y**2, axis=1)
        denom_x = 1 + 2 * beta * (1 + x_norms_squared)
        denom_y = 1 + 2 * beta * (1 + y_norms_squared)
        denominator = np.sqrt(np.outer(denom_x, denom_y))
        return alpha * (2.0/np.pi) * np.arcsin(numerator / denominator)
    return kern


class GaussianProcess:

    def __init__(self, kernel: Callable, noise: float):
        """
        Initialize a GP model with the specified kernel and noise
        :param kernel: the kernel to use when fitting the data/predicting
        :param noise: the sample noise assumed to be added to the data
        """
        # todo <your code here>
        self.kernel = kernel
        self.noise = noise

        self.x_train = None
        self.y_train = None
        self.L = None
        self.alpha = None

    def fit(self, X, y) -> 'GaussianProcess':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        # todo <your code here>
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the MMSE regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        # todo <your code here>
        return None

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def sample(self, X) -> np.ndarray:
        """
        Sample a function from the posterior
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the sample (same shape as X)
        """
        # todo <your code here>
        return None

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        # todo <your code here>
        return None

    def log_evidence(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the model's log-evidence under the training data
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the log-evidence of the model under the data points
        """
        self.fit(X, y)
        # todo <your code here>
        return None


def main():
    # ------------------------------------------------------ section 2.1
    xx = np.linspace(-5, 5, 500)
    x, y = np.array([-2, -1, 0, 1, 2]), np.array([2.4, .9, 2.8, -2.9, -1.5])

    # ------------------------------ questions 2 and 3
    # choose kernel parameters
    params = [
        # Laplacian kernels
        ['Laplacian', Laplacian_kernel, 1, 0.25],           # insert your parameters, order: alpha, beta
        ['Laplacian', Laplacian_kernel, None, None],        # insert your parameters, order: alpha, beta
        ['Laplacian', Laplacian_kernel, None, None],        # insert your parameters, order: alpha, beta

        # RBF kernels
        ['RBF', RBF_kernel, 1, 0.25],                       # insert your parameters, order: alpha, beta
        ['RBF', RBF_kernel, None, None],                    # insert your parameters, order: alpha, beta
        ['RBF', RBF_kernel, None, None],                    # insert your parameters, order: alpha, beta

        # Gibbs kernels
        ['Spectral', Spectral_kernel, 1, .5, 3],            # insert your parameters, order: alpha, beta, gamma
        ['Spectral', Spectral_kernel, None, None, None],    # insert your parameters, order: alpha, beta, gamma
        ['Spectral', Spectral_kernel, None, None, None],    # insert your parameters, order: alpha, beta, gamma

        # Neurel network kernels
        ['NN', NN_kernel, 1, 0.25],                         # insert your parameters, order: alpha, beta
        ['NN', NN_kernel, None, None],                      # insert your parameters, order: alpha, beta
        ['NN', NN_kernel, None, None],                      # insert your parameters, order: alpha, beta
    ]
    noise_var = 0.05

    # plot all of the chosen parameter settings
    for p in params:
        # create kernel according to parameters chosen
        k = p[1](*p[2:])    # p[1] is the kernel function while p[2:] are the kernel parameters

        # initialize GP with kernel defined above
        gp = GaussianProcess(k, noise_var)

        # plot prior variance and samples from the priors
        plt.figure()
        # todo <your code here>
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title(KERNEL_STRS[p[0]].format(*p[2:]))
        plt.ylim([-5, 5])

        # fit the GP to the data and calculate the posterior mean and confidence interval
        gp.fit(x, y)
        m, s = gp.predict(xx), 2*gp.predict_std(xx)

        # plot posterior mean, confidence intervals and samples from the posterior
        plt.figure()
        plt.fill_between(xx, m-s, m+s, alpha=.3)
        plt.plot(xx, m, lw=2)
        for i in range(6): plt.plot(xx, gp.sample(xx), lw=1)
        plt.scatter(x, y, 30, 'k', zorder=10)
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title(KERNEL_STRS[p[0]].format(*p[2:]))
        plt.ylim([-5, 5])
        plt.show()

    # ------------------------------ question 4
    # define range of betas
    betas = np.linspace(0.1, 7, 101)
    noise_var = .27

    # calculate the evidence for each of the kernels
    evidence = [GaussianProcess(RBF_kernel(1, beta=b), noise_var).log_evidence(x, y) for b in betas]

    # plot the evidence as a function of beta
    plt.figure()
    plt.plot(betas, evidence, lw=2)
    plt.xlabel(r'$\beta$')
    plt.ylabel('log-evidence')
    plt.show()

    # extract betas that had the min, median and max evidence
    srt = np.argsort(evidence)
    min_ev, median_ev, max_ev = betas[srt[0]], betas[srt[(len(evidence)+1)//2]], betas[srt[-1]]

    # plot the mean of the posterior of a GP using the extracted betas on top of the data
    plt.figure()
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=min_ev), noise_var).fit(x, y).predict(xx), lw=2, label='min evidence')
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=median_ev), noise_var).fit(x, y).predict(xx), lw=2, label='median evidence')
    plt.plot(xx, GaussianProcess(RBF_kernel(1, beta=max_ev), noise_var).fit(x, y).predict(xx), lw=2, label='max evidence')
    plt.scatter(x, y, 30, 'k', alpha=.5)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()



