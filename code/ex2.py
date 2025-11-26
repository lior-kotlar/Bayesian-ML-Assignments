import numpy as np
from matplotlib import pyplot as plt
from typing import Callable
from os import path
from ex2_utils import plot_predictions_gt

DATA_DIRECTORY = path.abspath('data/')

def polynomial_basis_functions(degree: int) -> Callable:
    """
    Create a function that calculates the polynomial basis functions up to (and including) a degree
    :param degree: the maximal degree of the polynomial basis functions
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             polynomial basis functions, a numpy array of shape [N, degree+1]
    """
    def pbf(x: np.ndarray):
        # todo <your code here>
        scaled_x = x / degree
        design_matrix = np.vander(scaled_x, degree + 1, increasing=True)
        return design_matrix
    
    return pbf


def fourier_basis_functions(num_freqs: int) -> Callable:
    """
    Create a function that calculates the fourier basis functions up to a certain frequency
    :param num_freqs: the number of frequencies to use
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             Fourier basis functions, a numpy array of shape [N, 2*num_freqs + 1]
    """
    def fbf(x: np.ndarray):
        # todo <your code here>
        return None
    return fbf


def spline_basis_functions(knots: np.ndarray) -> Callable:
    """
    Create a function that calculates the cubic regression spline basis functions around a set of knots
    :param knots: an array of knots that should be used by the spline
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             cubic regression spline basis functions, a numpy array of shape [N, len(knots)+4]
    """
    def csbf(x: np.ndarray):
        # todo <your code here>
        return None
    return csbf


def learn_prior(hours: np.ndarray, temps: np.ndarray, basis_func: Callable) -> tuple:
    """
    Learn a Gaussian prior using historic data
    :param hours: an array of vectors to be used as the 'X' data
    :param temps: a matrix of average daily temperatures in November, as loaded from 'jerus_daytemps.npy', with shape
                  [# years, # hours]
    :param basis_func: a function that returns the design matrix of the basis functions to be used
    :return: the mean and covariance of the learned covariance - the mean is an array with length dim while the
             covariance is a matrix with shape [dim, dim], where dim is the number of basis functions used
    """
    thetas = []
    # iterate over all past years
    for i, t in enumerate(temps):
        ln = LinearRegression(basis_func).fit(hours, t)
        # todo <your code here>
        thetas.append(None)  # append learned parameters here

    thetas = np.array(thetas)

    # take mean over parameters learned each year for the mean of the prior
    mu = np.mean(thetas, axis=0)
    # calculate empirical covariance over parameters learned each year for the covariance of the prior
    cov = (thetas - mu[None, :]).T @ (thetas - mu[None, :]) / thetas.shape[0]
    return mu, cov


class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray, theta_cov: np.ndarray, sig: float, basis_functions: Callable):
        """
        Initializes a Bayesian linear regression model
        :param theta_mean:          the mean of the prior
        :param theta_cov:           the covariance of the prior
        :param sig:                 the signal noise to use when fitting the model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        # todo <your code here>
        self.basis_functions = basis_functions
        self.prior_mean = theta_mean
        self.prior_cov = theta_cov
        self.sigma = sig

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        # todo <your code here>
        d_matrix = self.basis_functions(X)
        inv_post_cov = np.linalg.inv(np.linalg.inv(self.prior_cov) + (1 / self.sigma**2) * (d_matrix.T @ d_matrix))
        epsilon = 1e-6
        self.post_cov = np.linalg.inv(inv_post_cov + epsilon*np.eye(len(inv_post_cov)))
        self.post_mean = self.post_cov @ (np.linalg.inv(self.prior_cov) @ self.prior_mean + (1 / self.sigma**2) * (d_matrix.T @ y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using MMSE
        :param X: the samples to predict
        :return: the predictions for X
        """
        # todo <your code here>
        d_matrix = self.basis_functions(X)
        if hasattr(self, 'post_mean'):
            return d_matrix @ self.post_mean
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

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        # todo <your code here>
        return None

    def posterior_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model and sampling from the posterior
        :param X: the samples to predict
        :return: the predictions for X
        """
        # todo <your code here>
        return None


class LinearRegression:

    def __init__(self, basis_functions: Callable):
        """
        Initializes a linear regression model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        # todo <your code here>
        self.basis_functions = basis_functions
        self.theta = None
        

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the model to the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        # todo <your code here>
        d_matrix = self.basis_functions(X)
        self.theta = np.linalg.solve(d_matrix.T @ d_matrix, d_matrix.T @ y)
        # self.theta = np.linalg.pinv(d_matrix.T @ d_matrix) @ d_matrix.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        # todo <your code here>
        d_matrix = self.basis_functions(X)
        if self.theta is not None:
            return d_matrix @ self.theta
        
        raise ValueError("Model is not fitted yet.")

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and return the predicted values for X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)


def main():
    nov_16_path = path.join(DATA_DIRECTORY, 'nov162024.npy')
    # load the data for November 16 2024
    nov16_temperatures = np.load(nov_16_path)
    nov16_hours = np.arange(0, 24, .5)
    train_temperature = nov16_temperatures[:len(nov16_temperatures)//2]
    train_hours = nov16_hours[:len(nov16_temperatures)//2]
    test_temperatures = nov16_temperatures[len(nov16_temperatures)//2:]
    test_hours = nov16_hours[len(nov16_temperatures)//2:]

    # setup the model parameters
    degrees = [3, 7]

    # ----------------------------------------- Classical Linear Regression
    for d in degrees:
        ln = LinearRegression(polynomial_basis_functions(d)).fit(train_hours, train_temperature)

        # print average squared error performance
        predictions = ln.predict(test_hours)
        squared_errors = (test_temperatures - predictions) ** 2
        mse = np.mean(squared_errors)
        print(f'Average squared error with LR and d={d} is {mse:.2f}')
        
        # plot graphs for linear regression part
        # todo <your code here>
        plot_predictions_gt(hours=test_hours,
                            true_temps=test_temperatures,
                            predicted_temps=predictions,
                            save_path=path.join(DATA_DIRECTORY, f'true_vs_pred_temps_d{d}_lr.png'),
                            title=f'LR Predictions vs Ground Truth (d={d})')

    # ----------------------------------------- Bayesian Linear Regression

    jerus_daytemps_path = path.join(DATA_DIRECTORY, 'jerus_daytemps.npy')
    # load the historic data
    temps = np.load(jerus_daytemps_path).astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)

    # setup the model parameters
    sigma = 0.25
    degrees = [3, 7]  # polynomial basis functions degrees

    # frequencies for Fourier basis
    freqs = [1, 2, 3]

    # sets of knots K_1, K_2 and K_3 for the regression splines
    knots = [np.array([12]),
             np.array([8, 16]),
             np.array([6, 12, 18])]

    # ---------------------- polynomial basis functions
    for deg in degrees:
        pbf = polynomial_basis_functions(deg)
        mu, cov = learn_prior(hours, temps, pbf)

        blr = BayesianLinearRegression(mu, cov, sigma, pbf)

        # plot prior graphs
        # todo <your code here>

        # plot posterior graphs
        # todo <your code here>

    # ---------------------- Gaussian basis functions
    for ind, K in enumerate(freqs):
        rbf = fourier_basis_functions(K)
        mu, cov = learn_prior(hours, temps, rbf)

        blr = BayesianLinearRegression(mu, cov, sigma, rbf)

        # plot prior graphs
        # todo <your code here>

        # plot posterior graphs
        # todo <your code here>

    # ---------------------- cubic regression splines
    for ind, k in enumerate(knots):
        spline = spline_basis_functions(k)
        mu, cov = learn_prior(hours, temps, spline)

        blr = BayesianLinearRegression(mu, cov, sigma, spline)

        # plot prior graphs
        # todo <your code here>

        # plot posterior graphs
        # todo <your code here>


if __name__ == '__main__':
    main()
