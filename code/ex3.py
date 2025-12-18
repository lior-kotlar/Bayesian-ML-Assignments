import numpy as np
from matplotlib import pyplot as plt
from os import path
from ex3_utils import BayesianLinearRegression, polynomial_basis_functions, load_prior, INPUT_DATA_DIRECTORY, OUTPUT_DATA_DIRECTORY


def log_evidence(model: BayesianLinearRegression, X, y):
    """
    Calculate the log-evidence of some data under a given Bayesian linear regression model
    :param model: the BLR model whose evidence should be calculated
    :param X: the observed x values
    :param y: the observed responses (y values)
    :return: the log-evidence of the model on the observed data
    """
    # extract the variables of the prior distribution
    mu = model.mu
    prior_cov = model.cov
    sig_sq = model.sig
    prior_prec = model.prec

    # extract the variables of the posterior distribution
    model.fit(X, y)
    map = model.fit_mu
    map_cov = model.fit_cov

    # calculate the log-evidence
    # <your code here>
    d = len(mu)
    n = len(y)
    _, post_cov_logdet = np.linalg.slogdet(map_cov)
    norm_term = (d/2)*np.log(2*np.pi) + 0.5*post_cov_logdet
    
    y_pred = model.h(X) @ map
    residual = y - y_pred
    term_likelihood = -(n/2)*(np.log(2*np.pi) + np.log(sig_sq)) - np.sum(residual**2) / (2*sig_sq)

    _, prior_cov_logdet = np.linalg.slogdet(prior_cov)
    prior_term = -0.5 * (d*np.log(2*np.pi) + prior_cov_logdet + (map - mu).T @ prior_prec @ (map - mu))

    return term_likelihood + prior_term + norm_term

def plot_evidence_vs_degree(degrees, degree_evidences, function_index, x, y, alpha, noise_var):
    best_degree_idx = np.argmax(degree_evidences)
    best_degree = degrees[best_degree_idx]
    
    worst_degree_idx = np.argmin(degree_evidences)
    worst_degree = degrees[worst_degree_idx]

    x_plot = np.linspace(np.min(x), np.max(x), 1000)

    plt.figure(figsize=(8, 5))
    plt.plot(degrees, degree_evidences, marker='o', linestyle='-', linewidth=2, label='Log-Evidence')
    
    plt.scatter(best_degree, degree_evidences[best_degree_idx], color='red', s=150, marker='*', zorder=5, 
                label=f'Best Model (d={best_degree})')

    plt.title(f'Log-Evidence vs. Polynomial Degree (Function f{function_index+1})')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Log-Evidence Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot1_filename = path.join(OUTPUT_DATA_DIRECTORY, f'evidence_score_f{function_index+1}.png')
    plt.savefig(plot1_filename)
    plt.close()
    print(f'Saved {plot1_filename}')

    plt.figure(figsize=(10, 6))

    plt.scatter(x, y, s=15, color='gray', alpha=0.4, label='Noisy Data', zorder=1)

    def plot_model_predictions(degree, color, label_prefix, linestyle='-'):
        pbf = polynomial_basis_functions(degree)
        mean_prior, cov_prior = np.zeros(degree + 1), np.eye(degree + 1) * alpha
        
        model = BayesianLinearRegression(mean_prior, cov_prior, noise_var, pbf)
        model.fit(x, y)
        
        y_mean = model.predict(x_plot)
        y_std = model.predict_std(x_plot)
        
        plt.plot(x_plot, y_mean, color=color, linewidth=2, linestyle=linestyle, zorder=3,
                    label=f'{label_prefix} (d={degree})')
        plt.fill_between(x_plot, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2, zorder=2)

    plot_model_predictions(best_degree, color='blue', label_prefix='Best Model')

    if best_degree != worst_degree:
        plot_model_predictions(worst_degree, color='red', label_prefix='Worst Model', linestyle='--')

    plt.title(f'Best vs. Worst Model Fit (Function f{function_index+1})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(np.min(y)-2, np.max(y)+2)

    plot2_filename = path.join(OUTPUT_DATA_DIRECTORY, f'best_vs_worst_f{function_index+1}.png')
    plt.savefig(plot2_filename)
    plt.close()
    print(f'Saved {plot2_filename}\n' + '-'*30)


def main():
    # ------------------------------------------------------ section 2.1
    # set up the response functions
    f1 = lambda x: x ** 2 - 1
    f2 = lambda x: (-x ** 2 + 10 * x ** 3 + 50 * np.sin(x / 6) + 10) / 100
    f3 = lambda x: (.5 * x ** 6 - .75 * x ** 4 + 2.75 * x ** 2) / 50
    f4 = lambda x: 5 / (1 + np.exp(-4 * x)) - (x - 2 > 0) * x
    f5 = lambda x: 1 * (np.cos(x * 4) + 4 * np.abs(x - 2))
    functions = [f1, f2, f3, f4, f5]

    noise_var = .25
    x = np.linspace(-3, 3, 500)

    degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    alpha = 1

    # go over each response function and polynomial basis function
    for i, f in enumerate(functions):
        y = f(x) + np.sqrt(noise_var) * np.random.randn(len(x))
        degree_evidences = np.zeros(len(degrees))
        for j, d in enumerate(degrees):
            # set up model parameters
            pbf = polynomial_basis_functions(d)
            mean, cov = np.zeros(d + 1), np.eye(d + 1) * alpha

            # calculate evidence
            log_ev = log_evidence(BayesianLinearRegression(mean, cov, noise_var, pbf), x, y)
            # <your code here>
            degree_evidences[j] = log_ev

        # plot evidence versus degree and predicted fit
        # <your code here>
        plot_evidence_vs_degree(degrees, degree_evidences, i, x, y, alpha, noise_var)
        
    # ------------------------------------------------------ section 2.2
    # load relevant data
    nov16 = np.load(path.join(INPUT_DATA_DIRECTORY, 'nov162024.npy'))
    hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16) // 2]
    hours_train = hours[:len(nov16) // 2]

    # load prior parameters and set up basis functions
    mu, cov = load_prior()
    pbf = polynomial_basis_functions(7)

    noise_vars = np.linspace(.05, 2, 100)
    evs = np.zeros(noise_vars.shape)
    for i, n in enumerate(noise_vars):
        # calculate the evidence
        mdl = BayesianLinearRegression(mu, cov, n, pbf)
        log_ev = log_evidence(mdl, hours_train, train)
        # <your code here>

    # plot log-evidence versus amount of sample noise
    # <your code here>


if __name__ == '__main__':
    main()



