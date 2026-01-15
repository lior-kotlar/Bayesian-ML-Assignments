import numpy as np
from matplotlib import pyplot as plt
import sys
from ex5_utils import load_im_data, BayesianLinearRegression, gaussian_basis_functions, accuracy, Gaussian, plot_ims, OUTPUT_DIRECTORY

def get_decision_boundary(mu_pos, mu_neg):
    diff = mu_pos - mu_neg
    norm_diff = np.linalg.norm(mu_pos)**2 - np.linalg.norm(mu_neg)**2
    slope = -diff[0] / diff[1]
    intercept = norm_diff / (2 * diff[1])
    return slope, intercept

def section_1():
    # ------------------------------------------------------ section 1
    # define question variables
    sig, sig_0 = 0.1, 0.25
    mu_pos_prior, mu_neg_prior = np.array([1, 1]), np.array([-1, -1])

    # sample 5 points from each class
    np.random.seed(0)
    x_pos = np.array([.5, 0])[None, :] + np.sqrt(sig) * np.random.randn(5, 2)
    x_neg = np.array([-.5, -.5])[None, :] + np.sqrt(sig) * np.random.randn(5, 2)

    # <your code here>

    n = 5
    denominator = (n/sig) + (1/sig_0)

    post_cov = (1/denominator) * np.eye(2)
    mu_pos_post = (1/denominator)*((np.sum(x_pos, axis=0)/sig) + (mu_pos_prior/sig_0))
    mu_neg_post = (1/denominator) * ((np.sum(x_neg, axis=0)/sig) + (mu_neg_prior / sig_0))

    plt.figure(figsize=(8, 6))
    plt.scatter(x_pos[:, 0], x_pos[:, 1], label="Class (+)", c='blue')
    plt.scatter(x_neg[:, 0], x_neg[:, 1], label="Class (-)", c='red')
    x_vals = np.linspace(-2, 2, 100)
    m_mmse, c_mmse = get_decision_boundary(mu_pos_post, mu_neg_post)
    plt.plot(x_vals, m_mmse * x_vals + c_mmse, 'k-', linewidth=2, label="MMSE Boundary")

    for i in range(10):
        s_mu_p = np.random.multivariate_normal(mu_pos_post, post_cov)
        s_mu_m = np.random.multivariate_normal(mu_neg_post, post_cov)
        m_sample, c_sample = get_decision_boundary(s_mu_p, s_mu_m)
        if i == 0:
            plt.plot(x_vals, m_sample * x_vals + c_sample, 'k--', alpha=0.2, label="Sampled Boundaries")
        else:
            plt.plot(x_vals, m_sample * x_vals + c_sample, 'k--', alpha=0.2)

    plt.legend()
    plt.title("Bayesian Decision Boundaries: MMSE vs Samples")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid(True)
    plt.savefig(OUTPUT_DIRECTORY + 'ex5_section1.png', dpi=300)
    plt.close()

def section_2_1(dogs, dogs_t, frogs, frogs_t, train, labels, test, labels_t):
    

    # ------------------------------------------------------ section 2.1
    nus = [0, 1, 5, 10, 25, 50, 75, 100]
    train_score, test_score = np.zeros(len(nus)), np.zeros(len(nus))
    for i, nu in enumerate(nus):
        beta = .05 * nu
        print(f'QDA with nu={nu}', end='', flush=True)

        # <your code here>
        gaus_dogs = Gaussian(beta=beta, nu=nu)
        gaus_dogs = gaus_dogs.fit(dogs)

        gaus_frogs = Gaussian(beta=beta, nu=nu)
        gaus_frogs = gaus_frogs.fit(frogs)

        ll_dogs_train = gaus_dogs.log_likelihood(train)
        ll_frogs_train = gaus_frogs.log_likelihood(train)
        preds_train = np.where(ll_dogs_train > ll_frogs_train, 1, -1)
        train_score[i] = accuracy(preds_train, labels)

        ll_dogs_test = gaus_dogs.log_likelihood(test)
        ll_frogs_test = gaus_frogs.log_likelihood(test)

        preds_test = np.where(ll_dogs_test > ll_frogs_test, 1, -1)
        test_score[i] = accuracy(preds_test, labels_t)

        print(f': train={train_score[i]:.2f}, test={test_score[i]:.2f}', flush=True)

    plt.figure()
    plt.plot(nus, train_score, lw=2, label='train')
    plt.plot(nus, test_score, lw=2, label='test')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel(r'value of $\nu$')
    plt.savefig(OUTPUT_DIRECTORY + 'ex5_section2_1.png', dpi=300)
    plt.close()

def section_2_2(dogs, dogs_t, frogs, frogs_t, train, labels, test, labels_t):
    # ------------------------------------------------------ section 2.2
    # define question variables
    beta = .02
    sigma = .1
    Ms = [250, 500, 750, 1000, 2000, 3000, 5750]
    train_score, test_score = np.zeros(len(Ms)), np.zeros(len(Ms))

    blr = None
    for i, M in enumerate(Ms):
        print(f'Gaussian basis functions using {M} samples', end='', flush=True)

        # < your code here >
        centers = np.concatenate([dogs[:M], frogs[:M]], axis=0)
        basis_funcs = gaussian_basis_functions(centers, beta=beta)
        num_features = centers.shape[0]
        prior_mean = np.zeros(num_features)
        prior_cov = np.eye(num_features)

        blr = BayesianLinearRegression(
            theta_mean=prior_mean, 
            theta_cov=prior_cov, 
            sig=sigma, 
            basis_functions=basis_funcs
        )
        blr.fit(train, labels)
        preds_train_raw = blr.predict(train)
        preds_test_raw = blr.predict(test)
        acc_train = accuracy(preds_train_raw, labels)
        acc_test = accuracy(preds_test_raw, labels_t)
        train_score[i] = acc_train
        test_score[i] = acc_test
        print(f': train={train_score[i]:.2f}, test={test_score[i]:.2f}', flush=True)

    plt.figure()
    plt.plot(Ms, train_score, lw=2, label='train')
    plt.plot(Ms, test_score, lw=2, label='test')
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('# of samples')
    plt.xscale('log')
    plt.savefig(OUTPUT_DIRECTORY + 'ex5_section2_2.png', dpi=300)
    plt.close()

    # calculate how certain the model is about the predictions
    d = np.abs(blr.predict(dogs_t) / blr.predict_std(dogs_t))
    inds = np.argsort(d)
    # plot most and least confident points
    plot_ims(dogs_t[inds][:25], 'least confident')
    plot_ims(dogs_t[inds][-25:], 'most confident')

def main():
    if len(sys.argv) < 2:
        print('usage: python ex5.py <part>')
        return
    
    part = sys.argv[1]
    if part == '1':
        section_1()
        return
    
    # ------------------------------------------------------ section 2
    # load image data
    (dogs, dogs_t), (frogs, frogs_t) = load_im_data()
    # split into train and test sets
    train = np.concatenate([dogs, frogs], axis=0)
    labels = np.concatenate([np.ones(dogs.shape[0]), -np.ones(frogs.shape[0])])
    test = np.concatenate([dogs_t, frogs_t], axis=0)
    labels_t = np.concatenate([np.ones(dogs_t.shape[0]), -np.ones(frogs_t.shape[0])])
    
    if part == '2.1':
        section_2_1(dogs, dogs_t, frogs, frogs_t, train, labels, test, labels_t)
        return

    if part == '2.2':
        section_2_2(dogs, dogs_t, frogs, frogs_t, train, labels, test, labels_t)
        return
    


if __name__ == '__main__':
    main()







