'''
MultiLayer Classifier (aka, multilayer perceptron)
    w/ Particle Swarm-Like Optimization added to Hidden Layer (https://en.wikipedia.org/wiki/Particle_swarm_optimization)
- - - - - - - - - - - - - - - - - - - - - - - - - - - 

--- Functions ---
    - forward <-- get model outputs
    - loss <-- cost function
    - loss_grad <-- returns gradients
    - response <-- luce-choice rule (ie, softmax without exponentiation)
    - fit <-- trains model on a number of epochs
    - predict <-- gets class predictions
    - build_params <-- returns dictionary of weights
    - update_params <-- updates weights


--- Notes ---
    - implements sum-squared-error cost function
    - hidden activation function & derivative have to be provided in 'hps' dictionary (there are some available in the utils.py script)
    - Particle Swarm-Like Hidden Layer:
        * each hidden node updates according to:
            (1) the direction of it's gradients
            (2) the direction of the average of the gradients for all other hidden nodes
            (missing the "cognitive" term of PSO, but could be included with 'momentum')
'''
## external requirements
import numpy as np

def softmax(x):
    x -= np.max(x)
    return (np.exp(x).T / np.sum(np.exp(x),axis=1)).T


## "forward pass"
def forward(params, inputs, hps):
    hidden_act_raw = np.add(
        np.matmul(
            inputs,
            params['input']['hidden']['weights']
        ),
        params['input']['hidden']['bias']
    )

    hidden_act = hps['hidden_activation'](hidden_act_raw)

    output_act_raw = np.add(
        np.matmul(
            hidden_act,
            params['hidden']['output']['weights']
        ),
        params['hidden']['output']['bias'],
    )

    output_act = hps['output_activation'](output_act_raw)

    return [hidden_act_raw, hidden_act, output_act_raw, output_act]


## cost function (sum squared error)
def loss(params, inputs, targets, hps):
    return np.sum(
        np.square(
            np.subtract(
                forward(params, inputs, hps)[-1],
                targets
            )
        )
    ) / inputs.shape[0]


## backprop (for sum squared error cost function)
def loss_grad(params, inputs, targets, hps):
    hidden_act_raw, hidden_act, output_act_raw, output_act = forward(params, inputs, hps)

    ## gradients for decode layer ( chain rule on cost function )
    decode_grad = np.multiply(
        hps['output_activation_deriv'](output_act_raw),
        (2 * (output_act - targets))  / inputs.shape[0] # <-- deriv of cost function
    )

    ## gradients for decode weights
    decode_grad_w = np.matmul(
        hidden_act.T,
        decode_grad
    )

    ## gradients for decode bias
    decode_grad_b = decode_grad.sum(axis = 0, keepdims = True)

    # - - - - - - - - - -

    ## gradients for encode layer ( chain rule on hidden layer )
    encode_grad = np.multiply(
        hps['hidden_activation_deriv'](hidden_act_raw),
        np.matmul(
            decode_grad, 
            params['hidden']['output']['weights'].T
        )
    )

    ## gradients for encode weights
    encode_grad_w = np.matmul(
        inputs.T,
        encode_grad
    )

    ## gradients for encode bias
    encode_grad_b = encode_grad.sum(axis = 0, keepdims = True)

    return {
        'input': {
            'hidden': {
                'weights': encode_grad_w,
                'bias': encode_grad_b,
            }
        },
        'hidden': {
            'output': {
                'weights': decode_grad_w,
                'bias': decode_grad_b,
            }
        }
    }


## luce choice
def response(params, inputs, hps):
    return softmax(
        forward(params, inputs, hps)[-1]
    )

## build parameter dictionary
def build_params(num_features, num_hidden_nodes, num_classes, weight_range = [-.1, .1]):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    num_classes <-- number of categories in the dataset
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'input': {
            'hidden': {
                'weights': np.random.uniform(*weight_range, [num_features, num_hidden_nodes]),
                'bias': np.random.uniform(*weight_range, [1, num_hidden_nodes]),
            }
        },
        'hidden': {
            'output': {
                'weights': np.random.uniform(*weight_range, [num_hidden_nodes, num_classes]),
                'bias': np.random.uniform(*weight_range, [1, num_classes]),
            }
        }
    }

def build_params_xavier(num_features, num_hidden_nodes, num_classes):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    num_classes <-- number of categories in the dataset
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'input': {
            'hidden': {
                'weights': np.random.normal(0, 1, [num_features, num_hidden_nodes]) * np.sqrt(2 / (num_features + num_hidden_nodes)),
                'bias': np.zeros([1, num_hidden_nodes]),
            }
        },
        'hidden': {
            'output': {
                'weights': np.random.normal(0, 1, [num_hidden_nodes, num_classes]) * np.sqrt(2 / (num_hidden_nodes + num_classes)),
                'bias': np.zeros([1, num_classes]),
            }
        }
    }

## weight update
def update_params(params, gradients, lr_unit, lr_swarm):
    #__hidden layer (hidden weights get added PSO update)
    params['input']['hidden']['weights'] -= (lr_unit * gradients['input']['hidden']['weights']) + (lr_swarm * np.mean(gradients['input']['hidden']['weights'], axis = -1, keepdims = True)) # <-- particle swarm optimization term
    params['input']['hidden']['bias'] -= lr_unit * gradients['input']['hidden']['bias'] + (lr_swarm * np.mean(gradients['input']['hidden']['bias'], axis = -1, keepdims = True)) # <-- particle swarm optimization term

    #__output layer (update normally)
    params['hidden']['output']['weights'] -= lr_unit * gradients['hidden']['output']['weights']
    params['hidden']['output']['bias'] -= lr_unit * gradients['hidden']['output']['bias']

    return params


## fit to training set
def fit(params, inputs, targets, hps, training_epochs = 1, randomize_presentation = True):
    presentation_order = np.arange(inputs.shape[0])

    for e in range(training_epochs):
        if randomize_presentation == True: np.random.shuffle(presentation_order)
        
        for i in range(inputs.shape[0]):

            params = update_params(
                params, 
                loss_grad(params, inputs[i:i+1,:], targets[i:i+1,:], hps), # <-- returns gradients
                hps['learning_rate'],
                hps['learning_rate_swarm'],
            )

    return params

## predict
def predict(params, inputs):
    return np.argmax(
        forward(params, inputs, hps)[-1],
        axis = 1
    )

## - - - - - - - - - - - - - - - - - -
## RUN MODEL
## - - - - - - - - - - - - - - - - - -
if __name__ == '__main__':
    np.random.seed(0)

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

    labels = [
        # 'A','A','A','A', 'B','B','B','B', # <-- type 1
        # 'A','A','B','B', 'B','B','A','A', # <-- type 2
        'A','A','A','B', 'B','B','B','A', # <-- type 4
        # 'B','A','A','B', 'A','B','B','A', # <-- type 6
    ]

    categories = np.unique(labels)
    idx_map = {category: idx for category, idx in zip(categories, range(len(categories)))}
    labels_indexed = [idx_map[label] for label in labels]
    one_hot_targets = np.eye(len(categories))[labels_indexed]

    sigmoid = lambda x:  1 / (1 + np.exp(-x))
    sigmoid_deriv = lambda x:  sigmoid(x) * (1 - sigmoid(x))

    hps = {
        'learning_rate': .5,  # <-- learning rate
        'learning_rate_swarm': 1,
        'weight_range': [-.3, .3],  # <-- weight range
        'num_hidden_nodes': 4,

        'hidden_activation': sigmoid,
        'hidden_activation_deriv': sigmoid_deriv,

        'output_activation': sigmoid,
        'output_activation_deriv': sigmoid_deriv,
    }

    params = build_params(
        inputs.shape[1],  # <-- num features
        hps['num_hidden_nodes'],
        len(categories),
        weight_range = hps['weight_range']
    )

    num_training_epochs = 100
    params = fit(params, inputs, one_hot_targets, hps, training_epochs = num_training_epochs)
    p = predict(params, inputs)
    print(p)

