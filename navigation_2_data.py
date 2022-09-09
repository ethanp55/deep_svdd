from deepsvdd import DeepSVDD, Objectives, roc_auc_score
import numpy as np
import os
import pandas as pd
import pickle
from tensorflow import keras
from sklearn.model_selection import train_test_split
from utils import grab_image_data, ALL_COMBOS, PERCENTAGE_TO_TRY


# Format the data arrays to make sure they have the proper shape and type
def format_array(arry):
    converted_arry = np.array(arry)
    n_samples, img_dim = converted_arry.shape[0], converted_arry.shape[-1]
    converted_arry = converted_arry.reshape(n_samples, img_dim, img_dim, -1)

    return converted_arry


data_dir = 'navigation_2'

# Grab all of the data from the files within the current data directory
for i, file in enumerate(sorted(os.listdir(f'data/{data_dir}/'))):
    # Initialize different data arrays we need
    x, y = [], []
    file_name = file.title().lower()
    name = file_name.split('.')[0]
    df = pd.read_csv(f'data/{data_dir}/{file_name}', header=None)

    for j in range(len(df)):
        x.append(df.iloc[j, :])
        y.append(1.0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=53339)
    x_train, x_test, y_train, y_test = list(x_train), list(x_test), list(y_train), list(y_test)

    # Expand the test arrays using all of the other data files
    for k, new_file in enumerate(sorted(os.listdir(f'data/{data_dir}/'))):
        new_file_name = new_file.title().lower()
        new_name = new_file_name.split('.')[0]

        if new_name == name:
            continue

        df = pd.read_csv(f'data/{data_dir}/{new_file_name}', header=None)

        for j in range(len(df)):
            x_test.append(df.iloc[j, :])
            y_test.append(0.0)

    # Convert each row od data into an "image"
    for j in range(len(x_train)):
        image_data = grab_image_data(np.array(x_train[j]).reshape(1, -1))
        x_train[j] = image_data

    for j in range(len(x_test)):
        image_data = grab_image_data(np.array(x_test[j]).reshape(1, -1))
        x_test[j] = image_data

    x_train, x_test = format_array(x_train), format_array(x_test)
    y_train, y_test = np.array(y_train), np.array(y_test)

    # Make sure we have all of the data and the x and y test arrays have the same number of samples
    if len(x_train) == 0 or len(x_test) == 0 or len(y_test) == 0 or len(x_test) != len(y_test):
        raise Exception('Could not find data for training and/or testing')

    # Set up different hyperparameters to test - we will select the best combination
    best_params, best_auc = None, -np.inf
    np.random.shuffle(ALL_COMBOS)
    possible_combos = ALL_COMBOS[:int(PERCENTAGE_TO_TRY * len(ALL_COMBOS))]
    print(f'TRYING {len(possible_combos)} DIFFERENT HYPERPARAMETER COMBINATIONS')

    for nu, rep_dim, k, lr in possible_combos:
        # Initialize the network - same one the Deep SVDD authors used on the MNIST dataset
        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(8, (5, 5), padding='same', use_bias=False, input_shape=x_train.shape[1:]))
        model.add(keras.layers.LeakyReLU(1e-2))
        model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
        model.add(keras.layers.MaxPool2D())

        model.add(keras.layers.Conv2D(4, (5, 5), padding='same', use_bias=False))
        model.add(keras.layers.LeakyReLU(1e-2))
        model.add(keras.layers.BatchNormalization(epsilon=1e-4, trainable=False))
        model.add(keras.layers.MaxPool2D())

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(rep_dim, use_bias=False))

        # Train and test - the model, center, and radius will be saved throughout the training process
        svdd = DeepSVDD(model, representation_dim=rep_dim, objective=Objectives.SOFT_BOUNDARY)
        svdd.fit(x_train, x_test, y_test, f'{data_dir}/{name}', n_epochs=50, verbose=False)

        pred = svdd.predict(x_test)
        auc = roc_auc_score(y_test, -pred)

        if auc > best_auc:
            print(f'Updating best AUC score to {auc}')
            best_params, best_auc = (nu, rep_dim, k, lr), auc

            # Save the ROC AUC score
            with open(f'./scores/{data_dir}/{name}_roc_auc.txt', 'w') as f:
                f.write(f'{auc}')

            # Save the raw output from the network
            with open(f'./scores/{data_dir}/{name}_output.pickle', 'wb') as f:
                pickle.dump(-pred, f)

    print(f'RESULTS WHEN TRAINED ON {name.upper()}')
    print(f'Best params: {best_params}')
    print(f'Best AUC: {best_auc}')
