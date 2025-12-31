import os
import numpy as np
from typing import Callable
from matplotlib import pyplot as plt
from scipy.linalg import cho_solve, solve_triangular, cholesky
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
        self.x_train = X
        self.y_train = y
        K_noise = self.kernel(X, X) + self.noise * np.eye(len(X))
        self.L = cholesky(K_noise, lower=True)
        self.alpha = cho_solve((self.L, True), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the MMSE regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        k_star = self.kernel(self.x_train, X)
        mean_predictions = k_star.T @ self.alpha

        return mean_predictions

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
        k_ss = self.kernel(X, X)
        if self.x_train is None:
            mu = np.zeros(X.shape[0])
            cov = k_ss
        else:
            k_star = self.kernel(self.x_train, X)
            v = solve_triangular(self.L, k_star, lower=True)
            mu = k_star.T @ self.alpha
            cov = k_ss - v.T @ v
        
        cov += 1e-8 * np.eye(X.shape[0])
        l_sample = cholesky(cov, lower=True)
        z = np.random.normal(size=X.shape[0])
        return mu + l_sample @ z

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        if self.x_train is None:
            return np.sqrt(np.diagonal(self.kernel(X, X)))
        k_star = self.kernel(self.x_train, X)
        v = solve_triangular(self.L, k_star, lower=True)
        k_diag = np.diagonal(self.kernel(X, X))
        v_squared_sum = np.sum(v**2, axis=0)
        variance = k_diag - v_squared_sum
        variance = np.maximum(variance, 0)
        return np.sqrt(variance)

    def log_evidence(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the model's log-evidence under the training data
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the log-evidence of the model under the data points
        """
        self.fit(X, y)
        term1 = -0.5 * y.T @ self.alpha
        term2 = -np.sum(np.log(np.diagonal(self.L)))
        n = len(y)
        term3 = - (n / 2) * np.log(2 * np.pi)
        return term1 + term2 + term3


def main():
    # ------------------------------------------------------ section 2.1
    xx = np.linspace(-5, 5, 500)
    x, y = np.array([-2, -1, 0, 1, 2]), np.array([2.4, .9, 2.8, -2.9, -1.5])

    # ------------------------------ questions 2 and 3
    # choose kernel parameters
    params = [
        # Laplacian kernels
        ['Laplacian', Laplacian_kernel, 1, 0.025],           # insert your parameters, order: alpha, beta
        ['Laplacian', Laplacian_kernel, 1, 1.0],        
        ['Laplacian', Laplacian_kernel, 10, 1.0],        

        # RBF kernels
        ['RBF', RBF_kernel, 1, 0.025],                       # insert your parameters, order: alpha, beta
        ['RBF', RBF_kernel, 1, 1.0],        
        ['RBF', RBF_kernel, 10, 1.0],                    

        # Gibbs kernels
        ['Spectral', Spectral_kernel, 1, 0.005, 1],            # insert your parameters, order: alpha, beta, gamma
        ['Spectral', Spectral_kernel, 1, 0.05, 1],    
        ['Spectral', Spectral_kernel, 1, 0.5, 1],    

        # Neurel network kernels
        ['NN', NN_kernel, 0.5, 0.005],                         # insert your parameters, order: alpha, beta
        ['NN', NN_kernel, 0.5, 0.5],                      
        ['NN', NN_kernel, 0.5, 50],                     
    ]
    noise_var = 0.05

    if not os.path.exists('data/output/ex4'):
        os.makedirs('data/output/ex4')

    # plot all of the chosen parameter settings
    for i, p in enumerate(params):
        # create kernel according to parameters chosen
        k = p[1](*p[2:])    # p[1] is the kernel function while p[2:] are the kernel parameters

        # initialize GP with kernel defined above
        gp = GaussianProcess(k, noise_var)

        # plot prior variance and samples from the priors
        plt.figure()

        m_prior = np.zeros_like(xx)
        s_prior = 2*gp.predict_std(xx)

        plt.fill_between(xx, m_prior - s_prior, m_prior + s_prior, alpha=.3, color='blue')
        plt.plot(xx, m_prior, 'k--', lw=2)

        for i in range(5):
            plt.plot(xx, gp.sample(xx), lw=1)
        
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.title(KERNEL_STRS[p[0]].format(*p[2:]))
        plt.ylim([-5, 5])

        hyper_parameters_string = f'_a={p[2]}_b={p[3]}'
        if p[0] == 'Spectral':
            hyper_parameters_string += f'_g={p[4]}'

        filename_prior = f"data/output/ex4/{p[0]}_Prior{hyper_parameters_string}.png"
        plt.savefig(filename_prior)
        plt.close()

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
        
        filename_posterior = f"data/output/ex4/{p[0]}_Posterior{hyper_parameters_string}.png"
        plt.savefig(filename_posterior)
        plt.close()
    
    # ------------------------------ question 4
    # define range of betas
    betas = np.linspace(0.1, 7, 101)
    noise_var = .27
    alpha = 1.0

    # calculate the evidence for each of the kernels
    evidence = [GaussianProcess(RBF_kernel(alpha, beta=b), noise_var).log_evidence(x, y) for b in betas]

    # plot the evidence as a function of beta
    plt.figure()
    plt.plot(betas, evidence, lw=2)
    plt.xlabel(r'$\beta$')
    plt.ylabel('log-evidence')
    plt.title('Log-Marginal Likelihood vs Beta')
    best_beta_idx = np.argmax(evidence)
    plt.axvline(betas[best_beta_idx], color='r', linestyle='--', label=f'Best Beta: {betas[best_beta_idx]:.2f}')
    plt.legend()
    plt.savefig('data/output/ex4/Evidence_Score.png')

    # extract betas that had the min, median and max evidence
    srt = np.argsort(evidence)
    median_idx = srt[len(evidence)//2]
    selected_betas = {
        'Min Evidence': betas[srt[0]],
        'Median Evidence': betas[median_idx],
        'Max Evidence': betas[srt[-1]]
    }

    plt.figure(figsize=(12, 4))
    colors = {'Min Evidence': 'red', 'Median Evidence': 'orange', 'Max Evidence': 'green'}

    for label, beta in selected_betas.items():
        gp = GaussianProcess(RBF_kernel(alpha, beta=beta), noise_var).fit(x, y)
        m = gp.predict(xx)
        s = 2 * gp.predict_std(xx)

        plt.plot(xx, m, lw=2, label=f'{label} (Beta={beta:.2f})', color=colors[label])
        plt.fill_between(xx, m - s, m + s, alpha=0.1, color=colors[label])

    plt.scatter(x, y, 50, 'k', zorder=10, label='Data')

    # plot the mean of the posterior of a GP using the extracted betas on top of the data
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.title('Comparison of Best, Worst, and Median Betas')
    plt.legend()
    plt.ylim([-5, 5])
    plt.savefig('data/output/ex4/Evidence_Comparison.png')


if __name__ == '__main__':
    main()



