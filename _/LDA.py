'''
Implementation of LDA by hand, based on this tutorial by Sebastian Raschka: https://sebastianraschka.com/Articles/2014_python_lda.html
    - copying the class structure of scikit-learn's PCA func
'''
import numpy as np 

class LDA():
    def __init__(self):
        self.components = None

    def fit(self, data, labels): # <-- using covariance method
        label_set = np.unique(labels)
        class_means = np.array([data[labels == label,:].mean(axis = 0, keepdims = True) for label in label_set])
        class_cov_mats = np.array([np.cov(data[labels == label,:].T) for label in label_set]).sum(axis = 0)

        overall_means = data.mean(axis = 0, keepdims = True)
        # print(np.subtract(class_means[0],means))
        
        overall_scat_mats = np.array([
            data[labels == l,:].shape[0] * (class_means[l] - overall_means).T @ (class_means[l] - overall_means)
            for l in label_set
        ]).sum(axis = 0)

        ## calculate eigenvalues of matmul of within_class_variability(inv) & between_class_variability
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(class_cov_mats) @ overall_scat_mats)

        # sort components, largest to smallest
        idx_sort = np.flip(eig_vals.argsort()) # <-- get ordering of eigenvectors: largest to smallest
        self.components = eig_vecs[:,idx_sort]
        return self.components


    def transform(self, data, num_components):
        assert num_components <= self.components.shape[1], '\n\n\t! you\'re asking for too many components (should be <= num original variables in dataset)\n\n'
        return data @ self.components[:,:num_components]


if __name__ == '__main__':
    num_features = 3
    c1 = np.random.normal(-2,1,[50,num_features])
    labels_c1 = [0]*50

    c2 = np.random.normal(0,1,[50,num_features])
    labels_c2 = [1]*50
    
    c3 = np.random.normal(2,1,[50,num_features])
    labels_c3 = [2]*50

    data = np.concatenate([c1,c2,c3], axis = 0)
    labels = np.array(labels_c1 + labels_c2 + labels_c3)

    lda_class = LDA()
    components = lda_class.fit(data, labels)

    num_components = 2
    transformed_data = lda_class.transform(data, num_components)

    print(data.shape, '-->', transformed_data.shape)
