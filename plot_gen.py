import matplotlib.pyplot as plt
from nearest_neighbour import np, predictknn2, gensmallm, learnknn

def knn_mnist_q_2a(sample_sizes, k, repeats):
    data = np.load('mnist_all.npz')

    train1, train3, train4, train6 = data['train1'], data['train3'], data['train4'], data['train6']
    test1, test3, test4, test6 = data['test1'], data['test3'], data['test4'], data['test6']


    x_test = np.vstack([test1, test3, test4, test6])
    y_test = np.concatenate([
        1 * np.ones(test1.shape[0]),
        3 * np.ones(test3.shape[0]),
        4 * np.ones(test4.shape[0]),
        6 * np.ones(test6.shape[0])

    ])

    
    means, mins, maxs = [], [], []

    for sample_size in sample_sizes:
        errors = []
        for _ in range(repeats):
            x_train, y_train = gensmallm([train1, train3, train4, train6], [1, 3, 4, 6], sample_size)
            classifer = learnknn(k, x_train, y_train)
            y_test_preds = predictknn2(classifer, x_test).reshape(-1)
            errors.append(np.mean(y_test != y_test_preds))
        errors = np.array(errors)
        means.append(np.mean(errors))
        mins.append(np.min(errors))
        maxs.append(np.max(errors))
    return means, mins, maxs


def create_plot_2_a(sample_sizes, means, mins, maxs, plot_name):
    plt.errorbar(
        sample_sizes, 
        means, 
        yerr=[means - mins, maxs - means],
        fmt='-o',
        capsize=5,
        capthick=2,
        elinewidth=2
    )
    plt.xlabel('Training sample size')
    plt.ylabel('Average test error')
    plt.title('k-NN test error vs training size (k=1)')
    plt.xticks(sample_sizes)
    plt.savefig(f'{plot_name}.png', dpi=300, bbox_inches='tight')
    plt.show()


def q2_a_plot():
    sample_sizes = [1,5,10,20,50,100]
    means, mins, maxs = knn_mnist_q_2a(sample_sizes, 1, 10)
    create_plot_2_a(np.array(sample_sizes), np.array(means), np.array(mins), np.array(maxs), 'q2a')


def knn_mnist_q_2d(data, sample_size, k_vals, repeats):

    train1, train3, train4, train6 = data['train1'], data['train3'], data['train4'], data['train6']
    test1, test3, test4, test6 = data['test1'], data['test3'], data['test4'], data['test6']


    x_test = np.vstack([test1, test3, test4, test6])
    y_test = np.concatenate([
        1 * np.ones(test1.shape[0]),
        3 * np.ones(test3.shape[0]),
        4 * np.ones(test4.shape[0]),
        6 * np.ones(test6.shape[0])

    ])
    print(f"x_test shape: {x_test.shape}")  # verify size
    print(f"Starting m={sample_size}...")
    
    means = []

    for k in k_vals:
        print(f"  k={k}...")
        errors = []
        for _ in range(repeats):
            x_train, y_train = gensmallm([train1, train3, train4, train6], [1, 3, 4, 6], sample_size)
            classifer = learnknn(k, x_train, y_train)
            y_test_preds = predictknn2(classifer, x_test).reshape(-1)
            errors.append(np.mean(y_test != y_test_preds))
        errors = np.array(errors)
        means.append(np.mean(errors))
        print(f"  k={k} done, mean error={means[-1]:.4f}")
    return means


def create_plot_2_d(k_values, means, sample_size):
    plt.figure()
    plt.plot(k_values, means, '-o', markersize=8)
    plt.xlabel('k')
    plt.ylabel('Average test error')
    plt.title(f'k-NN test error vs training size (m={sample_size})')
    plt.xticks(k_values)
    plt.savefig(f'q2d_m_{sample_size}.png', dpi=300, bbox_inches='tight')
    plt.close()


def q2_d_plot():
    data = np.load('mnist_all.npz')
    sample_sizes = [100,400]
    k_vals = [1,3,5,7,9,11,13,15]
    for m in sample_sizes:
        means= knn_mnist_q_2d(data, m, k_vals, 25) 
        create_plot_2_d(np.array(k_vals), np.array(means), m)

def add_label_noise(y, noise_level, all_labels=[1, 3, 4, 6]):
    """Randomly corrupt a fraction of labels to a different label"""
    y_noisy = y.copy()
    n = len(y)
    num_corrupt = int(noise_level * n)
    corrupt_indices = np.random.choice(n, num_corrupt, replace=False)
    
    for i in corrupt_indices:
        other_labels = [l for l in all_labels if l != y_noisy[i]]
        y_noisy[i] = np.random.choice(other_labels)
    
    return y_noisy


def knn_mnist_q_2e(data, sample_size, k_vals, noise_levels, repeats):
    train1, train3, train4, train6 = data['train1'], data['train3'], data['train4'], data['train6']
    test1, test3, test4, test6 = data['test1'], data['test3'], data['test4'], data['test6']

    x_test = np.vstack([test1, test3, test4, test6])
    y_test = np.concatenate([
        1 * np.ones(test1.shape[0]),
        3 * np.ones(test3.shape[0]),
        4 * np.ones(test4.shape[0]),
        6 * np.ones(test6.shape[0])
    ])

    # results[k][noise_level] = mean error
    results = {k: [] for k in k_vals}

    for k in k_vals:
        print(f"  k={k}...")
        for noise in noise_levels:
            errors = []
            for _ in range(repeats):
                x_train, y_train = gensmallm([train1, train3, train4, train6], [1, 3, 4, 6], sample_size)
                # corrupt both train and test labels
                y_train_noisy = add_label_noise(y_train, noise)
                y_test_noisy = add_label_noise(y_test, noise)
                
                classifier = learnknn(k, x_train, y_train_noisy)
                y_test_preds = predictknn2(classifier, x_test).reshape(-1)
                errors.append(np.mean(y_test_noisy != y_test_preds))
            results[k].append(np.mean(errors))
            print(f"    noise={noise:.0%} done, mean error={results[k][-1]:.4f}")
    
    return results

def bayes_optimal_error(noise_level, num_classes=4):
    """
    With noise level p, a fraction p of labels are corrupted to one of 3 other labels.
    The Bayes optimal error is the probability that the noisy label differs from the true label.
    For uniform noise over 3 other classes:
    P(corrupted) = p * (3/4) ... wait, let's think carefully:
    - With prob (1-p): label is correct
    - With prob p: label is changed to one of 3 others uniformly
    So Bayes error = p * (1 - 1/3) ... 
    Actually the clean Bayes error is ~0 if we assume perfect separation,
    but realistically it's:
    bayes_error = p * (num_classes - 1) / num_classes
    """
    return noise_level * (num_classes - 1) / num_classes


def create_plot_2_e(noise_levels, results, k_vals, sample_size):
    plt.figure()
    
    for k in k_vals:
        plt.plot(noise_levels, results[k], '-o', markersize=6, label=f'k={k}')
    
    # Bayes optimal error
    bayes_errors = [n for n in noise_levels]
    plt.plot(noise_levels, bayes_errors, '--', color='red', label='Bayes optimal')
    
    plt.xlabel('Noise level')
    plt.ylabel('Average test error')
    plt.title(f'k-NN test error vs noise level (m={sample_size})')
    plt.xticks(noise_levels)
    plt.legend()
    plt.savefig(f'q2e_m_{sample_size}.png', dpi=300, bbox_inches='tight')
    plt.close()

def q2_e_plot():
    data = np.load('mnist_all.npz')
    sample_sizes = [50, 100, 400]
    k_vals = [1, 7, 15]
    noise_levels = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    repeats = 10

    for m in sample_sizes:
        print(f"Starting m={m}...")
        results = knn_mnist_q_2e(data, m, k_vals, noise_levels, repeats)
        create_plot_2_e(noise_levels, results, k_vals, m)


if __name__ == '__main__':
    q2_e_plot()
