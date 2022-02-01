'''
Attention Learning COVEring Map (Kruschke, 1992)
- - - - - - - - - - - - - - - - - - - - - - - - - - - 

--- Functions ---
    - forward <-- get model outputs
    - loss <-- cost function
    - loss_grad <-- returns gradients
    - response <-- luce-choice rule (ie, softmax without exponentiation)
    - focusing <-- biases impact of diverse dimensions during reconstruction
    - fit <-- trains model on a number of epochs
    - predict <-- gets class predictions
    - build_params <-- returns dictionary of weights
    - update_params <-- updates weights

--- Note ---
    - really need to update backprop to be a lot cleaner than it is
'''
## external requirements
import numpy as np

## minkowski pairwise distance function (https://en.wikipedia.org/wiki/Minkowski_distance)
def pdist(a1, a2, r, **kwargs):
    attention_weights = kwargs.get('attention_weights', np.ones([1,a1.shape[1]]) / a1.shape[1])

    # format inputs & exemplars for (i think vectorized) pairwise distance calculations
    a1_tiled = np.tile(a1, a2.shape[0]).reshape(a1.shape[0], a2.shape[0], a1.shape[1])
    a2_tiled = np.repeat([a2], a1.shape[0], axis=0)

    distances = np.power(
        np.sum(
            np.multiply(
                attention_weights,
                np.abs(a1_tiled - a2_tiled) ** r
            ),
            axis = 2,
        ),
        1/r
    )

    return distances


## "forward pass"
def forward(model, inputs, exemplars, c, r):

    distances = pdist(inputs, exemplars, r, attention_weights = params['attention_weights'])

    # exemplar layer activations
    hidden_activation = np.exp(
        (-c) * distances
    )

    # class predictions (luce-choiced)
    output_activation = np.matmul(
            hidden_activation,
            params['association_weights']
        )

    return [hidden_activation, output_activation]


## sum squared error loss function
def loss(params, inputs, exemplars, c, r, targets):
    activations = forward(params, inputs, exemplars, c, r)[-1]

    return .5 * np.sum(
        np.square(
            np.subtract(
                activations,
                targets
            )
        )
    ) / inputs.shape[0]


## get gradients
def loss_grad(params, inputs, exemplars, c, r, targets):

    hidden_activation, output_activation = forward(params, inputs, exemplars, c, r)    
    
    targets = (output_activation * targets).clip(1) * targets # <-- humble teacher principle (performs max(1,t) func on correct category labels, and min(-1,t) on incorrect channels)

    association_gradients = (targets - output_activation).T * hidden_activation # <-- this makes sense for the most part

    attention_gradients = -np.matmul( # <-- i have no clue what this whole operation is doing
        c * np.multiply(
            np.sum(
                    params['association_weights'] * np.repeat(
                        (targets - output_activation), 
                        exemplars.shape[0], 
                        axis = 0
                ),
                axis = 1
            ),
            hidden_activation
        ),
        np.abs(
            np.subtract(
                exemplars,
                np.repeat(
                    inputs,
                    exemplars.shape[0],
                    axis = 0
                )
            )
        )
    )

    return {
        'attention_weights': attention_gradients,
        'association_weights': association_gradients.T,
    }


def response(params, inputs, exemplars, c, r, phi): # softmax
    output_activation = forward(params, inputs, exemplars, c, r)[-1]
    return np.divide(
        np.exp(output_activation * phi),
      #---------#
        np.sum(
            np.exp(output_activation * phi), 
            axis=1, keepdims=True
        )
    )


def build_params(num_features, num_exemplars, num_categories):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_categories <-- (list) 
    '''
    return {
        'attention_weights': np.ones([1, num_features]) / num_features, # <-- input -to- hidden 
        'association_weights': np.zeros([num_exemplars, num_categories]), # <-- hidden -to- output
    }


def update_params(params, gradients, attention_lr, association_lr):
    params['attention_weights'] += attention_lr * gradients['attention_weights']
    params['attention_weights'] = params['attention_weights'] * (params['attention_weights'] > 0)

    params['association_weights'] += association_lr * gradients['association_weights']
    return params



def fit(params, inputs, exemplars, targets, c, r, attention_lr, association_lr, training_epochs = 1, randomize_presentation = True):
    presentation_order = np.arange(inputs.shape[0])

    for e in range(training_epochs):
        if randomize_presentation == True: np.random.shuffle(presentation_order)

        for i in presentation_order:        

            params = update_params(
                params, 
                loss_grad(params, inputs[i:i+1,:], exemplars, c, r, targets[i:i+1,:]), # <-- returns gradients
                attention_lr,
                association_lr,
            )

    return params


## predict
def predict(params, inputs, exemplars, c, r):
    return np.argmax(
        response(params, inputs, exemplars, c, r, 1),
        axis = 1
    )



## - - - - - - - - - - - - - - - - - -
## RUN MODEL
## - - - - - - - - - - - - - - - - - -
if __name__ == '__main__':
    # np.random.seed(0)

    inputs = np.array([
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [1, 0, 0],

        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],

    ])
    exemplars = inputs

    labels = [
        'A','A','A','A', 'B','B','B','B', # <-- type 1
        # 'A','A','B','B', 'B','B','A','A', # <-- type 2
        # 'A','A','A','B', 'B','B','B','A', # <-- type 4
        # 'B','A','A','B', 'A','B','B','A', # <-- type 6
    ]

    categories = np.unique(labels)
    idx_map = {category: idx for category, idx in zip(categories, range(len(categories)))}
    labels_indexed = [idx_map[label] for label in labels]
    one_hot_targets = np.eye(len(categories))[labels_indexed] * 2 - 1

    hps = {
        'association_lr': .1, # <-- association learning rate
        'attention_lr': .2, # <-- attention learning rate
        'c': 1, # <-- specificity
        'r': 1, # <-- distance metric (1: cityblock, 2: euclidean)
        'phi': 4, # <-- response mapping parameter
    }

    params = build_params(
        inputs.shape[1],  # <-- num features
        exemplars.shape[0],  # <-- num exemplars
        len(categories),
    )

    # p = forward(params, inputs, exemplars, hps['c'], hps['r'])[-1]

    num_training_epochs = 10

    params = fit(
        params, 
        inputs, 
        exemplars,
        one_hot_targets,
        hps['c'],
        hps['r'],
        hps['attention_lr'],
        hps['association_lr'],
        training_epochs = num_training_epochs,
        randomize_presentation = False
    )

    p = forward(params, inputs, exemplars, hps['c'], hps['r'])[-1]
    print(p)
