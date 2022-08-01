import enum
import tensorflow as tf
import numpy as np
from math import ceil
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


class Objectives(enum.Enum):
    SOFT_BOUNDARY = 'soft_boundary'
    ONE_CLASS = 'one_class'


class DeepSVDD:
    def __init__(self, model, objective=Objectives.ONE_CLASS, nu=0.1, representation_dim=32, batch_size=128,
                 lr=0.001, k=10):
        # Set up different variables we need
        self.representation_dim = representation_dim
        self.objective = objective
        self.model = model
        self.nu = nu
        initializer = tf.keras.initializers.GlorotUniform()
        r_shape, c_shape = [], [self.representation_dim]
        self.r = tf.Variable(name='r', shape=r_shape, dtype=tf.float32, trainable=False, initial_value=initializer(shape=r_shape))
        self.c = tf.Variable(name='c', shape=c_shape, dtype=tf.float32, trainable=False, initial_value=initializer(shape=c_shape))
        self.k = k
        self.batch_size = batch_size

        # Create optimizer
        self.opt = tf.keras.optimizers.Adam(lr)

    @tf.function
    def train(self, x):
        with tf.GradientTape() as tape:
            # Get the output from the network and calculate the loss
            output = self.model(x, training=True)
            dist = tf.reduce_sum(tf.square(output - self.c), axis=-1)

            if self.objective == Objectives.SOFT_BOUNDARY:
                score = dist - self.r ** 2
                loss = self.r ** 2 + (1 / self.nu) * tf.reduce_mean(tf.maximum(score, tf.zeros_like(score)))

            else:
                loss = tf.reduce_mean(dist)

        # Use the loss and gradients to update the model weights
        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_weights))

        return loss, dist

    def fit(self, x, x_test, y_test, n_epochs=10, verbose=True):
        # Get training set size and number of batches
        n = x.shape[0]
        num_batches = int(ceil(n / self.batch_size))

        # Calculate the center c variable
        self._init_c(x)

        # Iterate through epochs
        for epoch in range(n_epochs):
            epoch_loss = 0

            indices = np.random.permutation(n)
            x_train = x[indices]
            g_batch = tqdm(range(num_batches)) if verbose else range(num_batches)

            # Iterate through each batch
            for i_batch in g_batch:
                x_batch = x_train[i_batch * self.batch_size: (i_batch + 1) * self.batch_size]

                loss, dist = self.train(x_batch)
                epoch_loss += loss

                # Update the R variable
                if self.objective == Objectives.SOFT_BOUNDARY and epoch >= self.k:
                    self.r.assign(np.quantile(np.sqrt(dist), 1 - self.nu))

            if verbose:
                print(f'Epoch {epoch + 1} complete, total loss = {epoch_loss}, avg loss = {epoch_loss / n} -- Test '
                      f'results:')
                pred = self.predict(x_test)
                auc = roc_auc_score(y_test, -pred)
                print(f'ROC AUC score = {auc}')

    def predict(self, x):
        # Make predictions for a test set x - uses the score function defined in the paper
        n = x.shape[0]
        num_batches = int(ceil(n / self.batch_size))
        scores = list()

        for i in range(num_batches):
            batch_i = x[i * self.batch_size: (i + 1) * self.batch_size]
            output = self.model(batch_i, training=False)
            dist = tf.reduce_sum(tf.square(output - self.c), axis=-1)

            if self.objective == Objectives.SOFT_BOUNDARY:
                score = dist - self.r ** 2

            else:
                score = dist

            scores.append(score)

        return np.concatenate(scores)

    def _init_c(self, x, eps=0.1):
        # Initialize the center c variable - based on the paper and original code from the authors
        n = x.shape[0]
        num_batches = int(ceil(n / self.batch_size))
        latent_sum = None

        for i in range(num_batches):
            batch_i = x[i * self.batch_size: (i + 1) * self.batch_size]
            output = self.model(batch_i, training=False)

            if latent_sum is None:
                latent_sum = np.zeros(output.shape[-1])

            latent_sum += tf.reduce_sum(output, axis=0)

        c = latent_sum / n

        new_c = np.array(c)
        new_c[np.where(np.equal(True, (abs(c) < eps) & (c < 0)))] = -eps
        new_c[np.where(np.equal(True, (abs(c) < eps) & (c > 0)))] = eps

        self.c.assign(new_c)
