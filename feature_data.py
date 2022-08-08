from deepsvdd import DeepSVDD, Objectives, roc_auc_score
import numpy as np
import os
import pandas as pd
import pickle
from pyts.image import GramianAngularField


# Function that converts a single row of data into an "image"
def grab_image_data(subset):
    gasf_transformer = GramianAngularField(method='summation')
    gasf_subset = gasf_transformer.transform(subset)

    return gasf_subset


REPRESENTATION_DIM = 32

# Grab the feature change data and train a network for each data directory
for data_dir in sorted(os.listdir('data/feature_changes')):
    # Skip any hidden directories
    if '.' in data_dir:
        continue

    # Format the data arrays to make sure they have the proper shape and type
    def _format_array(arry):
        converted_arry = np.array(arry)
        n_samples, img_dim = converted_arry.shape[0], converted_arry.shape[-1]
        converted_arry = converted_arry.reshape(n_samples, img_dim, img_dim, -1)

        return converted_arry

    x_total, y_total = [], []

    # Iterate through the different data files in the directory
    for i, file in enumerate(sorted(os.listdir(f'data/feature_changes/{data_dir}/'))):
        file_name = file.title().lower()

        # Only look at files that have feature data
        if 'feature' in file_name:
            name = file_name.split('.')[0]
            feature_num = name.split('_')[-1]
            df = pd.read_csv(f'data/feature_changes/{data_dir}/{file_name}', header=None)
            labels = pd.read_csv(f'data/feature_changes/{data_dir}/y_{feature_num}.csv', header=None)
            x, y = [], []

            for j in range(len(df)):
                # Convert each row of data into an "image"
                image_data = grab_image_data(np.array(df.iloc[j, :]).reshape(1, -1))

                # Note that we do "1 - label" since the way we trained the model and the way the data are labelled
                # are reversed
                label = 1 - labels.iloc[j, 0]

                # Build up the data arrays
                x.append(image_data)
                y.append(label)
                x_total.append(image_data)
                y_total.append(label)

            x, y = _format_array(x), np.array(y)

            # Make sure we have all of the data and the x and y test arrays have the same number of samples
            if len(x) == 0 or len(x) != len(y):
                raise Exception('Could not find data for training and/or testing')

            # Initialize the network - used the saved models obtained from running robot_data.py
            svdd = DeepSVDD(None, representation_dim=REPRESENTATION_DIM, objective=Objectives.SOFT_BOUNDARY,
                            file_path=f'./saved_models/{data_dir}')
            pred = svdd.predict(x)
            auc = roc_auc_score(y, -pred)

            print(f'ROC AUC FOR {name}: {auc}')

            # Save the ROC AUC score
            with open(f'./scores/{data_dir}/{name}_roc_auc.txt', 'w') as f:
                f.write(f'{auc}')

            # Save the raw output from the network
            with open(f'./scores/{data_dir}/{name}_output.pickle', 'wb') as f:
                pickle.dump(-pred, f)

    # Get ROC AUC for entire directory (i.e. all the files in the directory)
    x_total, y_total = _format_array(x_total), np.array(y_total)

    # Make sure we have all of the data and the x and y test arrays have the same number of samples
    if len(x_total) == 0 or len(x_total) != len(y_total):
        raise Exception('Could not find data for training and/or testing')

    # Initialize the network - used the saved models obtained from running robot_data.py
    svdd = DeepSVDD(None, representation_dim=REPRESENTATION_DIM, objective=Objectives.SOFT_BOUNDARY,
                    file_path=f'./saved_models/{data_dir}')
    pred = svdd.predict(x_total)
    auc = roc_auc_score(y_total, -pred)

    print(f'ROC AUC FOR ENTIRE DIRECTORY {data_dir}: {auc}')

    # Save the ROC AUC score
    with open(f'./scores/{data_dir}/overall_roc_auc.txt', 'w') as f:
        f.write(f'{auc}')

    # Save the raw output from the network
    with open(f'./scores/{data_dir}/overall_output.pickle', 'wb') as f:
        pickle.dump(-pred, f)
