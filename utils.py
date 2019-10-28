import numpy as np

## - - - - - - - - - - - - - - - - - - - - - - -
# Convenience Functions
## - - - - - - - - - - - - - - -

def organize_data_from_txt(data_filepath, delimiter = ','):
    data = np.genfromtxt(data_filepath, delimiter = delimiter)

    data = {
        'inputs': data[:,:-1],
        'labels': data[:,-1],
        'categories': np.unique(data[:,-1]),
    }

    # map categories to label indices
    data['idx_map'] = {category: idx for category, idx in zip(data['categories'], range(len(data['categories'])))}

    # map original labels to label indices
    data['labels_indexed'] = [data['idx_map'][label] for label in data['labels']]

    # generate one hot targets
    data['one_hot_targets'] = np.eye(len(data['categories']))[data['labels_indexed']]

    return data