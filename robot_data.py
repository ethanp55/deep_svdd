from deepsvdd import DeepSVDD, Objectives, roc_auc_score
import numpy as np
import os
import pandas as pd
from tensorflow import keras
from utils import grab_image_data, ALL_COMBOS, PERCENTAGE_TO_TRY


# Grab the data and train a network for each data directory
for data_dir in sorted(os.listdir('data/')):
    # Skip any hidden or feature data directories
    if '.' in data_dir or 'feature' in data_dir or data_dir == 'navigation_2':
        continue

    # Initialize different data arrays we need
    x_train, x_test, y_test = [], [], []

    # Grab all of the data from the files within the current data directory
    for i, file in enumerate(sorted(os.listdir(f'data/{data_dir}/'))):
        file_name = file.title().lower()
        name = file_name.split('.')[0]
        df = pd.read_csv(f'data/{data_dir}/{file_name}', header=None)

        for j in range(len(df)):
            # Convert each row of data into an "image"
            image_data = grab_image_data(np.array(df.iloc[j, :]).reshape(1, -1))

            # Build up the data arrays
            if 'normal' and 'train' in name:
                x_train.append(image_data)

            else:
                x_test.append(image_data)

                label = 0.0 if 'ab' in name else 1.0
                y_test.append(label)

    # Format the data arrays to make sure they have the proper shape and type
    def _format_array(arry):
        converted_arry = np.array(arry)
        n_samples, img_dim = converted_arry.shape[0], converted_arry.shape[-1]
        converted_arry = converted_arry.reshape(n_samples, img_dim, img_dim, -1)

        return converted_arry

    x_train, x_test = _format_array(x_train), _format_array(x_test)
    y_test = np.array(y_test)

    # Make sure we have all of the data and the x and y test arrays have the same number of samples
    if len(x_train) == 0 or len(x_test) == 0 or len(y_test) == 0 or len(x_test) != len(y_test):
        raise Exception('Could not find data for training and/or testing')

    # Set up different hyperparameters to test - we will select the best one
    best_params, best_auc = None, -np.inf
    np.random.shuffle(ALL_COMBOS)
    possible_combos = ALL_COMBOS[:int(PERCENTAGE_TO_TRY * len(ALL_COMBOS))]

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
        svdd = DeepSVDD(model, representation_dim=rep_dim, objective=Objectives.SOFT_BOUNDARY, nu=nu, lr=lr, k=k)
        svdd.fit(x_train, x_test, y_test, data_dir, n_epochs=50, verbose=True)

        pred = svdd.predict(x_test)
        auc = roc_auc_score(y_test, -pred)

        if auc > best_auc:
            best_params, best_auc = (nu, rep_dim, k, lr), auc

    print(f'Best params: {best_params}')
    print(f'Best AUC: {best_auc}')

