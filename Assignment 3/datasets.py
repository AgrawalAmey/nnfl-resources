import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons, \
                             make_blobs, make_classification, make_blobs
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

def generate_datsets():
    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============

    np.random.seed(0)

    n_samples = 1500

    datasets = {}

    datasets['noisy_circles'] = make_circles(n_samples=n_samples, factor=.5,
                                             noise=.05)
    datasets['noisy_moons'] = make_moons(n_samples=n_samples, noise=.05)
    datasets['blobs'] = make_blobs(n_samples=n_samples, random_state=8)
    datasets['linearly_separable'] = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                                                         n_informative=2, random_state=1, n_clusters_per_class=1)
    # Anisotropicly distributed data
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    datasets['aniso'] = (X_aniso, y)

    # blobs with varied variances
    datasets['varied'] = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    
    return datasets

def plot_datasets(datasets):
    # ============
    # Set up cluster parameters
    # ============
    plt.figure(figsize=(2.3 * len(datasets) - 0.3, 4.3))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)

    dataset_num = 1

    for name, dataset in datasets.items():
        X = dataset[0]
        Ys = dataset[1:]

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
        
        for i in range(len(Ys)):
            plt.subplot(2, len(datasets), dataset_num + i * len(datasets))

            if i == 0:
                plt.title(name)

            colors = np.array(list(islice(cycle(flatui), int(max(Ys[i]) + 1))))
            plt.scatter(X[:, 0], X[:, 1], s=5, c=colors[Ys[i]])            
            
            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())

        dataset_num += 1

    plt.show()