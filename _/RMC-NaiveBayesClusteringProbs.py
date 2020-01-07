'''
DISCLAIMER: There are probably some issues with this code.
    ~ It seems to behave as you would expect on a problem like IRIS (high coupling probability leads to bigger prototypes; low coupling probability leads to exemplar model)
    ~ ^ but it's a bit different from Anderson's model (and i probably did some things wrong), so it's behavior shouldn't reflect the theory (which, IMO, is a pretty cool theory)

Rational Model of Categorization (Anderson 1990-91) w/ Naive Bayes Cluster Probabilities (Only makes sense for continuous features)
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
# np.seterr('raise')
import matplotlib.pyplot as plt 

g = 20
param_space = [
    np.linspace(0,1,g),
    np.linspace(0,1,g),
]
mesh = np.array(np.meshgrid(*param_space)).reshape(2,g*g).T

def gaussian_kernelv(x, mean, var): # <-- annoying how this doesn't sum to 1
    if 0 in var: var = np.full([1,x.shape[1]],.1)
    return np.multiply(
        (1 / np.sqrt(2 * np.pi * var)),
        np.exp(- ((x - mean) ** 2) / (2 * var))
    )


# RMC
c = .5 # <-- coupling probability
def train_rmc(inputs, clusters = None):
    categories = np.unique(inputs[:,-1])

    if clusters == None: clusters = np.array([0] + [np.nan] * (inputs.shape[0]-1))

    N = 1
    K = [0]
    p_k = [(c * np.equal(clusters[:1], 0).sum()) / ((1-c) + (c * N)) for k in K] # <-- custer baserates
    p_0 = (1 - c) / ((1 - c) + (c * N))

    p_F__k = np.array([
        gaussian_kernelv(
            inputs[0:1,:-1],
            inputs[:1][np.equal(clusters[:1], 0), :-1].mean(axis = 0, keepdims = True),
            inputs[:1][np.equal(clusters[:1], 0), :-1].var(axis = 0, keepdims = True)
            # inputs[:1][np.equal(clusters[:1], k), :-1].var(axis = 0, keepdims = True),
        )
        for k in K
    ])

    p_F__k = np.product(p_F__k, axis = -1)
    p_k__F = (p_k * p_F__k) / np.sum(p_k * p_F__k, axis = 0)

    # p_j__k = [np.sum(inputs[clusters == 0,-1] == category) / inputs[clusters == 0].shape[0] for category in categories]


    for i in range(1,inputs.shape[0]):
        N = i + 1

        K = np.unique(clusters[:i])
        p_k = [(c * np.equal(clusters[:i], 0).sum()) / ((1-c) + (c * N)) for k in K] # <-- custer baserates
        p_0 = (1 - c) / ((1 - c) + (c * N))

        p_F__k = np.array([
            gaussian_kernelv(
                inputs[i:i+1,:-1],
                inputs[:i][np.equal(clusters[:i], k), :-1].mean(axis = 0, keepdims = True),
                inputs[:i][np.equal(clusters[:i], k), :-1].var(axis = 0, keepdims = True),
            )
            for k in K
        ])

        p_F__k = np.product(p_F__k.clip(.0001, 99999), axis = -1)
        p_k__F = (p_k @ p_F__k) / np.sum(p_k @ p_F__k, axis = 0)
        # print(p_k__F)
        if p_F__k.max() < p_0:
            best = max(clusters) + 1 # <-- new cluster
        else:
            best = p_F__k.argmax()
        clusters[i] = best
        # p_j__k = [np.sum(inputs[clusters == best,-1] == category) / inputs[clusters == best].shape[0] for category in categories]
        # response_probs  = p_k__F * p_j__k

    response_probs = []
    N = inputs.shape[0]
    for i in range(inputs.shape[0]):

        K = np.unique(clusters)
        p_k = [(c * np.equal(clusters, 0).sum()) / ((1-c) + (c * N)) for k in K] # <-- custer baserates
        p_0 = (1 - c) / ((1 - c) + (c * N))

        p_F__k = np.array([
            gaussian_kernelv(
                inputs[i:i+1,:-1],
                inputs[np.equal(clusters, k), :-1].mean(axis = 0, keepdims = True),
                inputs[np.equal(clusters, k), :-1].var(axis = 0, keepdims = True),
            )
            for k in K
        ])

        p_F__k = np.product(p_F__k, axis = -1)
        p_k__F = (p_k @ p_F__k) / np.sum(p_k @ p_F__k, axis = 0)

        if p_F__k.max() < p_0:
            best = max(clusters) + 1 # <-- new cluster
        else:
            best = p_F__k.argmax()
        clusters[i] = best
        p_j__k = [np.sum(inputs[clusters == best,-1] == category) / inputs[clusters == best].shape[0] for category in categories]
        response_probs.append(p_k__F * p_j__k)

    return np.array(response_probs), clusters









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
    # data = np.genfromtxt('iris.csv', delimiter = ',')
    # categories = np.unique(data[:,-1])

    data = np.array([
        [.1, .4, 0],
        [.2, .3, 0],
        [.3, .2, 0],
        [.4, .1, 0],

        [.6, .9, 1],
        [.7, .8, 1],
        [.8, .7, 1],
        [.9, .6, 1],
    ])
    categories = np.unique(data[:,-1])

    probs, clusters = train_rmc(
        data, clusters = None
    )

    print(probs)


    # ##__Plot Results
    # import matplotlib.pyplot as plt 
    # from matplotlib.gridspec import GridSpec

    # fig = plt.figure(
    #     # figsize = [4,2]
    # )
    # gs = GridSpec(1,5)

    # ax_ = plt.subplot(gs[:,:4])
    # ax_.imshow(
    #     probs,
    #     cmap = 'binary', aspect = 'auto', vmin = 0, vmax = 1
    # )
    # ax_.set_xticks(range(len(categories)))
    # ax_.set_xticklabels(categories)

    # ax_.set_yticks(range(data.shape[0]))
    # ax_.set_yticklabels([
    #     ' ' if categories[probs[item,:].argmax()] == data[item,-1] else 'x'
    #     for item in range(data.shape[0])
    # ])

    # ax_.set_ylabel('items\n(x = incorrect prediction)\n')
    # ax_.set_title('Class Probabilities')


    # ax_ = plt.subplot(gs[:,4])
    # ax_.set_title('Cluster Probs')
    # ax_.scatter(
    #     x = [.5] * data.shape[0],
    #     y = np.linspace(0,1,data.shape[0]),
    #     c = clusters,
    #     cmap = 'tab20',
    #     alpha = .5
    # )
    # ax_.set_yticks([]); ax_.set_xticks([])

    # plt.savefig('test.png')
