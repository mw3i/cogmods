'''
MultiLayer Classifier (aka, multilayer perceptron)
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
    - only works with:
        - sigmoid activation function on the hidden layer
        - linear outputs
'''
## external requirements
import numpy as np

## Activation Functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_d(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

## "forward pass"
def forward(params, inputs):
    hidden_act_raw = np.add(
        np.matmul(
            inputs,
            params['input']['hidden']['weights']
        ),
        params['input']['hidden']['bias']
    )

    hidden_act = sigmoid(hidden_act_raw)

    output_act_raw = np.add(
        np.matmul(
            hidden_act,
            params['hidden']['output']['weights']
        ),
        params['hidden']['output']['bias'],
    )

    output_act = sigmoid(output_act_raw)

    return [hidden_act_raw, hidden_act, output_act_raw, output_act]


## cost function (sum squared error)
def loss(params, inputs, targets):
    return np.sum(
        np.square(
            np.subtract(
                forward(params, inputs)[-1],
                targets
            )
        )
    ) / inputs.shape[0]


## backprop (for sum squared error cost function)
def loss_grad(params, inputs, targets):
    hidden_act_raw, hidden_act, output_act_raw, output_act = forward(params, inputs = inputs)

    glob_err = (2 * (output_act - targets))

    ## gradients for channel weights
    decode_grad_w = np.multiply(
        hidden_act.T,
        np.multiply(
            sigmoid_d(output_act_raw),
            glob_err
        )
    )

    ## gradients for channel bias
    decode_grad_b = np.multiply(
        1, # <-- bias value
        np.multiply(
            sigmoid_d(output_act_raw),
            glob_err
        )
    )
    ## gradients for encode weights
    encode_grad_w = np.multiply(
        inputs.T,
        np.multiply(
            sigmoid_d(hidden_act_raw),
            np.matmul(
                np.multiply(
                    sigmoid_d(output_act_raw),
                    glob_err
                ), 
                params['hidden']['output']['weights'].T
            )
        )
    )

    ## gradients for encode bias
    encode_grad_b = np.multiply(
        1, # <-- bias value
        np.multiply(
            sigmoid_d(hidden_act_raw),
            np.matmul(
                np.multiply(
                    sigmoid_d(output_act_raw),
                    glob_err
                ), 
                params['hidden']['output']['weights'].T
            )
        )
    )

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
def response(params, inputs):
    return utils.softmax(
        forward(params, inputs)[-1]
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


## weight update
def update_params(params, gradients, lr):
    for layer in params:
        for connection in gradients[layer]:
            params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
            params[layer][connection]['bias'] -= lr * gradients[layer][connection]['bias']
    return params


## fit to training set
def fit(params, inputs, targets, learning_rate, training_epochs = 1, randomize_presentation = True):
    presentation_order = np.arange(inputs.shape[0])

    for e in range(training_epochs):
        if randomize_presentation == True: np.random.shuffle(presentation_order)
        
        for i in range(inputs.shape[0]):
            update_params(
                params, 
                loss_grad(params, inputs[i:i+1,:], targets[i:i+1,:]), # <-- returns gradients
                learning_rate,
            )

## predict
def predict(params, inputs):
    return np.argmax(
        forward(params, inputs)[-1],
        axis = 1
    )


## - - - - - - - - - - - - - - - - - -
## RUN MODEL
## - - - - - - - - - - - - - - - - - -
if __name__ == '__main__':
    # np.random.seed(0)

    inputs = np.array([
        [.1, .1],
        [.15, .15],
        [.2, .2],
        
        [.25, .25],
        [.3, .3],
        [.35, .35],
        
        [.4, .4],
        [.45, .45],
        [.5, .5],
        
        [.55, .55],
        [.6, .6],
        [.65, .65],
    ])

    gens = np.array([
        [.1, .1],
        [.15, .15],
        [.2, .2],
        
        [.25, .25],
        [.3, .3],
        [.35, .35],
        
        [.4, .4],
        [.45, .45],
        [.5, .5],
        
        [.55, .55],
        [.6, .6],
        [.65, .65],
        
        [.7, .7],
        [.75, .75],
        [.8, .8],
    ])

    labels = [
        'A','A','A','B','B','B','A','A','A','B','B','B', # <-- type 1
    ]


    categories = np.unique(labels)
    idx_map = {category: idx for category, idx in zip(categories, range(len(categories)))}
    labels_indexed = [idx_map[label] for label in labels]
    one_hot_targets = np.eye(len(categories))[labels_indexed]
    
    hps = {
        'lr': .05,  # <-- learning rate
        'wr': [-3, 3],  # <-- weight range
        'num_hidden_nodes': 4,
    }

    params = build_params(
        inputs.shape[1],  # <-- num features
        hps['num_hidden_nodes'],
        len(categories),
        weight_range = hps['wr'],
    )

    num_training_epochs = 100
    f = fit(params, inputs, one_hot_targets, hps['lr'], training_epochs = num_training_epochs)
    p = predict(params, gens)
    print(p)

