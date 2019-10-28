'''
MultiLayer Classifier (aka, multilayer perceptron)
- - - - - - - - - - - - - - - - - - - - - - - - - - - 

  ** SOM EDITION **
    + self organizing map, using hidden activation similarity 

  ** MAIN **
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
'''
## external requirements
import numpy as np

# dist_func = lambda x: np.exp(- ((x) ** 2))

push_strength = 2
pull_strength = 1

push_breadth = 20
pull_breadth = 10

dist_func = lambda x: np.subtract(
    pull_strength * np.exp(- pull_breadth * (x ** 2)), # <-- pull
    push_strength * np.exp(- push_breadth * (x ** 2)) # <-- push
)

# import matplotlib.pyplot as plt 
# plt.plot(
#     np.linspace(-10,10,100),
#     dist_func(np.linspace(-10,10,100))
# )

# plt.savefig('test.png')

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
def update_params(params, gradients, hid_activations, lr, social_gravity_strength = .1):

    vals = np.zeros(params['input']['hidden']['weights'].shape)

    for h_col, h in enumerate(hid_activations.T):
        for hh_col, hh in enumerate(hid_activations.T):
            vals[:,h_col] += dist_func(np.sum(h - hh)) * params['input']['hidden']['weights'][:,h_col]
    # ^^ i think this can be vectorized or asynched

    params['input']['hidden']['weights'] -= social_gravity_strength * vals # <-- add social impact

    ## update based on gradients
    params['input']['hidden']['weights'] -= lr * gradients['input']['hidden']['weights']
    params['input']['hidden']['bias'] -= lr * gradients['input']['hidden']['bias']

    params['hidden']['output']['weights'] -= lr * gradients['hidden']['output']['weights']
    params['hidden']['output']['bias'] -= lr * gradients['hidden']['output']['bias']

    # -----------------

    return params


## fit to training set
def fit(params, inputs, targets, hps, training_epochs = 1, randomize_presentation = True):
    presentation_order = np.arange(inputs.shape[0])

    for e in range(training_epochs):
        if randomize_presentation == True: np.random.shuffle(presentation_order)
        
        # for i in range(inputs.shape[0]):
        hid_activations = forward(params, inputs, hps)[1]
        params = update_params(
            params, 
            loss_grad(params, inputs, targets, hps), # <-- returns gradients
            hid_activations,
            hps['learning_rate'],
            social_gravity_strength = hps['social_gravity_strength'],
        )

    return params

## predict
def predict(params, inputs, hps):
    return np.argmax(
        forward(params, inputs, hps)[-1],
        axis = 1
    )

## - - - - - - - - - - - - - - - - - -
## RUN MODEL
## - - - - - - - - - - - - - - - - - -
if __name__ == '__main__':
    import copy

    import matplotlib.pyplot as plt

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
        'learning_rate': 1.5,  # <-- learning rate
        'weight_range': [-3, 3],  # <-- weight range
        'num_hidden_nodes': 20,

        # som part
        'social_gravity_strength': .05, # <-- strength of the social nudge in an organized direction
        # 'social_momentum': .2, # <-- ???

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

    num_epochs = 20
    params = fit(params, inputs, one_hot_targets, hps, training_epochs = num_epochs)
    p = predict(params, inputs, hps)

    print('original class labels: ', np.array(labels_indexed))
    print('predicted class labels:', p)
