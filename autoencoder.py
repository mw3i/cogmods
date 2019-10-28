'''
Autoencoder
- - - - - - - - - - - - - - - - - - - - - - - - - - - 

--- Functions ---
    - forward <-- get model outputs
    - loss <-- cost function
    - loss_grad <-- returns gradients
    - fit <-- trains model on a number of epochs
    - build_params <-- returns dictionary of weights
    - update_params <-- updates weights


--- Notes ---
    - implements sum-squared-error cost function
    - hidden activation function & derivative have to be provided in 'hps' dictionary (there are some available in the utils.py script)
'''

## external requirements
import numpy as np


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
def loss(params, inputs, hps, targets = None):
    if np.any(targets) == None: targets = inputs
    return np.sum(
        np.square(
            np.subtract(
                forward(params, inputs, hps)[-1],
                targets
            )
        )
    ) / inputs.shape[0]


## cost function (sum squared error)
def loss(params, inputs, hps, targets = None):
    if np.any(targets) == None: targets = inputs
    return np.sum(
        np.square(
            np.subtract(
                forward(params, inputs, hps)[-1],
                targets
            )
        )
    ) / inputs.shape[0]


## backprop (for sum squared error cost function)
def loss_grad(params, inputs, hps, targets = None):
    if np.any(targets) == None: targets = inputs

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

    # - - - - - - - - - - - -

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


## build parameter dictionary
def build_params(num_features, num_hidden_nodes, weight_range = [-.1, .1]):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    num_categories <-- number of category channels to make
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
                'weights': np.random.uniform(*weight_range, [num_hidden_nodes, num_features]),
                'bias': np.random.uniform(*weight_range, [1, num_features]),
            }
        }
    }

def build_params_xavier(num_features, num_hidden_nodes):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    num_categories <-- number of category channels to make
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
                'weights': np.random.normal(0, 1, [num_hidden_nodes, num_features]) * np.sqrt(2 / (num_hidden_nodes + num_features)),
                'bias': np.zeros([1, num_features]),
            }
        }
    }

## weight update
def update_params(params, gradients, lr):
    for layer in params:
        for connection in gradients[layer]:
            params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
            params[layer][connection]['bias'] -= lr * gradients[layer][connection]['bias']
    return params


## fit to training set
def fit(params, inputs, hps, targets = None, training_epochs = 1, randomize_presentation = True):
    if np.any(targets) == None: targets = inputs
    presentation_order = np.arange(inputs.shape[0])

    for e in range(training_epochs):
        if randomize_presentation == True: np.random.shuffle(presentation_order)
        
        for i in range(inputs.shape[0]):

            params = update_params(
                params, 
                loss_grad(params, inputs[i:i+1,:], hps, targets = targets[i:i+1,:]), # <-- returns gradients
                hps['learning_rate'],
            )

    return params


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

    sigmoid = lambda x:  1 / (1 + np.exp(-x))
    sigmoid_deriv = lambda x:  sigmoid(x) * (1 - sigmoid(x))

    hps = {
        'learning_rate': .05,  # <-- learning rate
        'weight_range': [-3, 3],  # <-- weight range
        'num_hidden_nodes': 4,

        'hidden_activation': sigmoid,
        'hidden_activation_deriv': sigmoid_deriv,

        'output_activation': lambda x: x, # <-- linear output function
        'output_activation_deriv': lambda x: 1, # <-- derivative of linear output function
    }

    params = build_params(
        inputs.shape[1],  # <-- num features
        hps['num_hidden_nodes'],
        weight_range = hps['weight_range']
    )

    num_training_epochs = 1000
    params = fit(params, inputs, hps, targets = inputs, training_epochs = num_training_epochs)
    p = forward(params, inputs, hps)[-1]
    print(p)

