from deepsvdd import DeepSVDD, Objectives, roc_auc_score
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from utils import grab_image_data, ALL_COMBOS, PERCENTAGE_TO_TRY

TRAIN_ON = ['normal', 'vision']

for train_on in TRAIN_ON:
    # Initialize different data arrays we need
    x_train, x_test, y_test, y = [], [], [], []

    # Grab all of the data from the files within the current data directory
    for i, file in enumerate(sorted(os.listdir(f'data/sawyer/'))):
        file_name = file.title().lower()
        name = file_name.split('.')[0]
        df = pd.read_csv(f'data/sawyer/{file_name}', header=None)

        for j in range(len(df)):
            # Convert each row of data into an "image"
            image_data = grab_image_data(np.array(df.iloc[j, :]).reshape(1, -1))

            # Build up the data arrays
            if train_on in name and 'test' not in name:
                x_train.append(image_data)
                y.append(1.0)

            else:
                x_test.append(image_data)

                label = 1.0 if train_on in name else 0.0
                y_test.append(label)

    if 'vision' in train_on:
        x_train, new_x_test, _, new_y_test = train_test_split(x_train, y, train_size=0.75, random_state=53339)
        x_test.extend(new_x_test)
        y_test.extend(new_y_test)

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
        svdd = DeepSVDD(model, representation_dim=rep_dim, objective=Objectives.SOFT_BOUNDARY, nu=nu, lr=lr, k=k)
        svdd.fit(x_train, x_test, y_test, f'sawyer_{train_on}', n_epochs=50, verbose=False)

        pred = svdd.predict(x_test)
        auc = roc_auc_score(y_test, -pred)

        if auc > best_auc:
            print(f'Updating best AUC score to {auc}')
            best_params, best_auc = (nu, rep_dim, k, lr), auc

    print(f'RESULTS WHEN TRAINED ON {train_on.upper()}')
    print(f'Best params: {best_params}')
    print(f'Best AUC: {best_auc}')

