'''
Implementation of PCA by hand, based on this tutorial by Sebastian Raschka: https://sebastianraschka.com/Articles/2014_pca_step_by_step.html
    - copying the class structure of scikit-learn
'''
import numpy as np 

class PCA():
    def __init__(self):
        self.components = None

    def fit(self, data): # <-- using covariance method
        cov_d = np.cov(data.T) # <-- get the covariance matrix

        ## calculate eigenvalues of the covariance matrix
        eig_val, eig_vec = np.linalg.eig(cov_d)

        # sort components, largest to smallest
        idx_sort = np.flip(eig_val.argsort()) # <-- get ordering of eigenvectors: largest to smallest
        self.components = eig_vec[:,idx_sort]
        return self.components

    def transform(self, data, num_components):
        assert num_components <= self.components.shape[1], '\n\n\t! you\'re asking for too many components (should be <= num original variables in dataset)\n\n'
        return data @ self.components[:,:num_components]


if __name__ == '__main__':
    data = np.random.normal(0,1,[10,3])

    pca_class = PCA()
    components = pca_class.fit(data)

    num_components = 2
    transformed_data = pca_class.transform(data, num_components)

    print(data.shape, '-->', transformed_data.shape)