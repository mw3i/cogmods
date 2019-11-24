'''
Naive Bayes Classifier
- - - - - - - - - - - - - - - - - - - - - - - - - - - 

--- Functions ---
    - predict <-- gets class predictions

    p(A|B) = p(B|A) * p(A) / p(B)

    based on this tutorial: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
'''

import numpy as np 

def gaussian_kernelv(x, data_mean, data_std):
    exponent = np.exp(- ((x - data_mean) ** 2 / (2 * data_std ** 2) ))
    return (1 / (np.sqrt(2 * np.pi) * data_std) * exponent)

# naive_bayes probability
def predict(inputs, data, labels):
    c = {}
    for category in categories:
        c[category] = {
            'data': data[labels == category],
        }
        c[category]['base_rate'] = c[category]['data'].shape[0] / data.shape[0]

    class_probabilities = []

    for category in categories:
        class_probabilities.append(
            np.multiply(
                gaussian_kernelv(inputs, c[category]['data'].mean(axis = 0), c[category]['data'].std(axis = 0)),
                c[category]['base_rate']
            )
        )

    class_probabilities = np.product(np.array(class_probabilities), axis = -1).T
    return class_probabilities / class_probabilities.max(axis = 1, keepdims = True) # <-- luce choice


if __name__ == '__main__':
    data = np.genfromtxt('iris.csv', delimiter = ',',dtype = float)[:,:-1]
    labels = np.genfromtxt('iris.csv', delimiter = ',', dtype = str)[:,-1]
    categories = np.unique(labels)

    probs = predict(
        # input data, reference data, reference labels
        data, data, labels
    )



    # ##__Plot Results
    # import matplotlib.pyplot as plt 

    # ax = plt.subplot()
    # ax.imshow(
    #     probs,
    #     cmap = 'binary', aspect = 'auto', vmin = 0, vmax = 1
    # )
    # ax.set_xticks(range(len(categories)))
    # ax.set_xticklabels(categories)

    # ax.set_yticks(range(data.shape[0]))
    # ax.set_yticklabels([
    #     ' ' if categories[probs[item,:].argmax()] == labels[item] else 'x'
    #     for item in range(data.shape[0])
    # ])
    # ax.set_ylabel('items\n(x = incorrect prediction)\n')
    
    # plt.title('Class Probabilities')
    # plt.show()
