from matplotlib.lines import Line2D
import numpy as np
from matplotlib import pyplot as plt
from typing import Callable
from os import path
from ex2_utils import plot_predictions_gt, DATA_DIRECTORY

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
        theta_vec = ln.theta
        thetas.append(theta_vec)  # append learned parameters here

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
        current_mean = self.prior_mean if not hasattr(self, 'post_mean') else self.post_mean
        pred_mean = d_matrix @ current_mean
        epistemic_variance = np.diag(d_matrix @ (self.prior_cov if not hasattr(self, 'post_cov') else self.post_cov) @ d_matrix.T)

        total_variance = epistemic_variance + (self.sigma ** 2)

        pred_std = np.sqrt(total_variance)
        
        return pred_mean, pred_std

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

def plot_predictions_for_bayesian_model(
        train_hours: np.ndarray,
        train_temps: np.ndarray,
        predict_hours: np.ndarray,
        predicted_mean_temps: np.ndarray,
        pred_std: np.ndarray,
        title: str,
        save_path: str,
        model: BayesianLinearRegression,
        num_samples: int = 5
    ):
    """    
    Args:
    train_hours (np.array of shape (hours,)): The input hours used for training.
    train_temps (np.array of shape (number of years, hours)): the observed temperatures used for training.
    predict_hours (np.array of shape (hours,)): The input hours for which predictions are made.
    predicted_mean_temps (np.array of shape (hours,)): The mean predicted temperatures from the model.
    pred_std (np.array of shape (hours,)): The standard deviation of the predicted temperatures.
    title (str): Title for the graph.
    save_path (str): Path to save the plotted graph.
    model: BayesianLinearRegression, the Bayesian linear regression model.
    num_samples (int): Number of prior samples to plot.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_hours, train_temps[0], color='gray', alpha=0.3, linewidth=1, label='Historical Data', zorder=1)
    for i in range(1, len(train_temps)):
        plt.plot(train_hours, train_temps[i], color='gray', alpha=0.3, linewidth=1, label='Historical Data', zorder=1)
    
    plt.plot(predict_hours, predicted_mean_temps, color='black', linewidth=3, label='Prior Mean', zorder=3)

    plt.fill_between(predict_hours,
                     predicted_mean_temps - pred_std, 
                     predicted_mean_temps + pred_std, 
                     color='blue', alpha=0.2, label='Confidence Interval (1$\sigma$)', zorder=2)
    
    jitter = 1e-6 * np.eye(len(model.prior_cov))
    safe_cov = model.prior_cov + jitter
    sampled_thetas = np.random.multivariate_normal(model.prior_mean, safe_cov, num_samples)
    Phi_dense = model.basis_functions(predict_hours)
    for i, theta_sample in enumerate(sampled_thetas):
        # Calculate curve: y = Phi * theta
        y_sample = Phi_dense @ theta_sample
        plt.plot(predict_hours, y_sample, linestyle='--', linewidth=1, alpha=0.8, label=f'Prior Sample {i+1}', zorder=4)


    # 5. Styling
    plt.title(title)
    plt.xlabel('Time of Day (Hours)')
    plt.ylabel('Temperature')
    plt.xlim(0, 24)
    # Use a smart legend that doesn't show every single history line
    handles, labels = plt.gca().get_legend_handles_labels()
    # Filter out duplicate labels if necessary, though Matplotlib usually handles unique labels well
    by_label = dict(zip(labels, handles))
    if 'Historical Data' in by_label:
        by_label['Historical Data'] = Line2D([0], [0], color='gray', linewidth=1)
        
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize='small')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path)
    
    # plt.show()

def classical_linear_regression(train_hours, train_temperature, test_hours, test_temperatures, degrees):
    for d in degrees:
        ln = LinearRegression(polynomial_basis_functions(d)).fit(train_hours, train_temperature)

        # print average squared error performance
        pred_mean = ln.predict(test_hours)
        squared_errors = (test_temperatures - pred_mean) ** 2
        mse = np.mean(squared_errors)
        print(f'Average squared error with LR and d={d} is {mse:.2f}')
        
        # plot graphs for linear regression part
        # todo <your code here>
        plot_predictions_gt(hours=test_hours,
                            true_temps=test_temperatures,
                            predicted_temps=pred_mean,
                            save_path=path.join(DATA_DIRECTORY, f'true_vs_pred_temps_d{d}_lr.png'),
                            title=f'LR Predictions vs Ground Truth (d={d})\nMSE={mse:.2f}')

def bayesian_linear_regression(degrees, hours, temps, sigma):
    for deg in degrees:
        pbf = polynomial_basis_functions(deg)
        mu, cov = learn_prior(hours, temps, pbf)

        blr = BayesianLinearRegression(mu, cov, sigma, pbf)

        # plot prior graphs
        hours_to_predict = np.arange(0, 24, 0.1)
        pred_mean, pred_std = blr.predict(hours_to_predict)
        plot_predictions_for_bayesian_model(
            train_hours=hours,
            train_temps=temps,
            predict_hours=hours_to_predict,
            predicted_mean_temps=pred_mean,
            pred_std=pred_std,
            deg=deg,
            title=f'Bayesian LR - Prior Predictions (Degree={deg})',
            save_path=path.join(DATA_DIRECTORY, f'bayesian_lr_prior_deg{deg}.png'),
            model=blr,
        )


        # plot posterior graphs
        # todo <your code here>

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
    classical_linear_regression(train_hours, train_temperature, test_hours, test_temperatures, degrees)

    # ----------------------------------------- Bayesian Linear Regression

    jerus_daytemps_path = path.join(DATA_DIRECTORY, 'jerus_daytemps.npy')
    # load the historic data
    temps = np.load(jerus_daytemps_path).astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)

    # setup the model parameters
    sigma = 0.25
    # frequencies for Fourier basis
    freqs = [1, 2, 3]
    # sets of knots K_1, K_2 and K_3 for the regression splines
    knots = [np.array([12]),
             np.array([8, 16]),
             np.array([6, 12, 18])]

    # ---------------------- polynomial basis functions
    bayesian_linear_regression(degrees, hours, temps, sigma)

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
