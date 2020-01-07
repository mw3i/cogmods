'''
!!! INCOMPLETE !!!

Rational Model of Categorization (Anderson 1990-91)
- - - - - - - - - - - - - - - - - - - - - - - - - - - 

Basic Idea:
    + a category label is a feature to be predicted by category-specific generative models of the data
    + the feature can be predicted optimially by combining:
        (a) the liklihood of an object belonging to a cluster given it's features
        (b) the liklihood of an object having some feature given the cluster (the predicted feature in this case is the category label)

    P(j|F) = P(k|F) * P(j|k)

        for all K clusters

    P(k|F) is the posterior probability that an object belongs to a cluster (relative to all other clusters). This is given by a luce-choice over the probilities of all clusters:

        P(k|F) = p(k) * p(F|k) / sum( p(k) * p(F|k) for all k in K )

            p(k) * p(F|k) <-- that part is naive bayes i think

    p(k) is given by the equation:

        p(k) = c * n_k / (1 - c) + cn

            n_k: number of items in partition k
            n: total number of items
            
            ^ so essentially, i think that those combined give us a baserate of a given cluster

            as Anderson (1991) puts it: this creates a strong "bias to put new items into [already existing] large categories"

    The probability that something is asigned to a new category is:
        P(0) = (1 - c) / ( (1-c) + cn )

    p(F|k) = product( P(j|k) for j in F)

    Notes:
        - the coupling parameter C tries to manipulate the liklihood that exemplars will be "grouped" into a cluster
            - when C is 0: each example gets it's own cluster
        - wtf is going on this the discrete -vs- continuous thing?


'''
import numpy as np; np.set_printoptions(linewidth = 10000)

import matplotlib.pyplot as plt 

g = 20
param_space = [
    np.linspace(0,1,g),
    np.linspace(0,1,g),
]
mesh = np.array(np.meshgrid(*param_space)).reshape(2,g*g).T

def gaussian_kernelv(x, data_mean, data_std):
    exponent = np.exp(- ((x - data_mean) ** 2 / (2 * data_std ** 2) ))
    return (1 / (np.sqrt(2 * np.pi) * data_std) * exponent)


# RMC
def predict_rmc(inputs, data, labels):
    # we want: P(j|F) <-- ie, given a set of features (the stimulus), what is the liklihood it has a feature j? (j being the category label)
    # we need: p(k|F) and p(j|k)


    # just going to assume we already know the clusters (lets say C is zero and each exemplar is it's own cluster)
    p_j__k = None
    p_k__F = None

    # get prob k given f: prob_k__F
    p_k = np.array([1 / data.shape[0] for cluster in data]) # <-- since each examplar has an equal base rate (im kind of cheating here)
    p_F__k = np.array([gaussian_kernelv(inputs, cluster, .1) for cluster in data])
    p_F__k = np.product(p_F__k, axis = -1)
    p_k__F = (p_F__k * p_k) / np.sum(p_F__k * p_k, axis = 0)

    # get prob j given k: p_j__k <-- i think this is just the category association weights
    p_j__k = np.zeros([inputs.shape[0], len(np.unique(labels))])
    p_j__k[labels == 'Iris-setosa',0] = 1
    p_j__k[labels == 'Iris-versicolor',1] = 1
    p_j__k[labels == 'Iris-virginica',2] = 1


    # combine them:
    response_probs = p_k__F @ p_j__k # <-- there's something wrong here, since the sum of the probabilities dont add to 1

    return response_probs









# naive_bayes probability
def predict_nb(inputs, data, labels):
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
    
    # data = np.array([
    #     [.1, .4],
    #     [.2, .3],
    #     [.3, .2],
    #     [.4, .1],

    #     [.6, .9],
    #     [.7, .8],
    #     [.8, .7],
    #     [.9, .6],
    # ])

    # labels = [
    #     0,0,0,0, 1,1,1,1, 
    # ]

    # categories = np.unique(labels)

    probs = predict_rmc(
        data, data, labels
    )

    # probs = predict_nb(
    #     # input data, reference data, reference labels
    #     data, data, labels
    # )
    

    ##__Plot Results
    import matplotlib.pyplot as plt 

    fig, ax = plt.subplots(
        1,1, 
        # figsize = [4,2]
    )
    ax.imshow(
        probs,
        cmap = 'binary', aspect = 'auto', vmin = 0, vmax = 1
    )
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)

    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels([
        ' ' if categories[probs[item,:].argmax()] == labels[item] else 'x'
        for item in range(data.shape[0])
    ])
    ax.set_ylabel('items\n(x = incorrect prediction)\n')
    ax.set_title('Class Probabilities')

    plt.savefig('test.png')



    # ##__Plot Results
    # import matplotlib.pyplot as plt 

    # fig, [ax,ax2] = plt.subplots(1,2, figsize = [4,2])

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
    # ax.set_title('Class Probabilities')


    # ax2.scatter(
    #     *data.T,
    #     c = labels
    # )

    # plt.tight_layout()
    # plt.savefig('test.png')
