import numpy as np
from matplotlib import pyplot as plt
from os import path
OUTPUT_DATA_DIRECTORY = path.abspath('data/output/ex2')
INPUT_DATA_DIRECTORY = path.abspath('data/input/ex2')

def plot_predictions_gt(hours: np.ndarray, true_temps: np.ndarray, predicted_temps: np.ndarray, save_path: str, title: str = 'Predictions vs Ground Truth'):
    """
    Plots true temperatures and predicted temperatures on the same graph.
    
    Args:
        hours (np.array): The time points (x-axis).
        true_temps (np.array): The ground truth temperatures.
        predicted_temps (np.array): The model's predictions.
        title (str): Title for the graph.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot True Data:
    # We use 'o' (dots) to show specific data points and a solid line to show continuity.
    # alpha=0.6 makes it slightly transparent so it doesn't dominate the prediction.
    plt.plot(hours, true_temps, 'o-', color='blue', label='True Temperature', alpha=0.6, markersize=5)
    
    # Plot Predicted Data:
    # We use a dashed line ('--') and a distinct color (red or orange) to differentiate it.
    # linewidth=2 makes the prediction stand out.
    plt.plot(hours, predicted_temps, '--', color='darkorange', label='Predicted Temperature', linewidth=2)
    
    # Labeling
    plt.xlabel('Time of Day (Hours)')
    plt.ylabel('Temperature')
    plt.title(title)
    plt.legend()
    
    # Grid and limits make the graph easier to read technically
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(hours.min(), hours.max())
    plt.savefig(save_path)
    # plt.show()

def temps_example():
    """
    An example of how to load the temperature data. Note that there are 2 data sets here
    """
    X = np.load(path.join(INPUT_DATA_DIRECTORY, 'jerus_daytemps.npy'))
    hours = [2, 5, 8, 11, 14, 17, 20, 23]
    plt.figure()
    for i in range(X.shape[0]):
        plt.plot(hours, X[i, :], lw=2)
    plt.title('Daily Average Temperatures in November')
    plt.xlabel('hour')
    plt.ylabel('temperature [C]')

    y = np.load(path.join(INPUT_DATA_DIRECTORY, 'nov162024.npy'))
    hours = np.arange(0, 24, .5)
    plt.figure()
    plt.plot(hours, y, lw=2)
    plt.title('Temperatures on November 16 2024')
    plt.xlabel('hour')
    plt.ylabel('temperature [C]')
    plt.show()


def confidence_interval_example():
    """
    An example of how random samples from the prior can be displayed, along with the mean function and confidence
    intervals
    """
    x = np.linspace(0, 2, 100)

    # create design matrix for 3rd order polynomial
    H = np.concatenate([x[:, None]**i for i in range(4)], axis=1)

    # create random prior
    mu = np.array([0, -3, 0, 1]) + .25*np.random.randn(4)
    S = .5*np.random.randn(H.shape[1], 100)
    S = S@S.T/100 + np.eye(mu.shape[0])*.001
    chol = np.linalg.cholesky(S)

    # find mean function
    mean = (H@mu[:, None])[:, 0]
    std = np.sqrt(np.diagonal(H@S@H.T))

    # plot mean with confidence intervals
    plt.figure()
    plt.fill_between(x, mean-std, mean+std, alpha=.5, label='confidence interval')
    for i in range(5):
        rand = (H@(mu[:, None] + chol@np.random.randn(chol.shape[-1], 1)))[:, 0]
        plt.plot(x, rand)
    plt.plot(x, mean, 'k', lw=2, label='mean')
    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x)$')
    plt.xlim([0, 2])
    plt.show()


if __name__ == '__main__':
    temps_example()
    confidence_interval_example()
