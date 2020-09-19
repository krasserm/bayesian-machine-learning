import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ------------------------------------------
#  Data
# ------------------------------------------


def select_bands(x, y, mask):
    assert x.shape[0] == y.shape[0]

    num_bands = len(mask)

    if x.shape[0] % num_bands != 0:
        raise ValueError('size of first dimension must be a multiple of mask length')

    data_mask = np.repeat(mask, x.shape[0] // num_bands)
    return [arr[data_mask] for arr in (x, y)]


def select_subset(x, y, num, rng=np.random):
    assert x.shape[0] == y.shape[0]

    choices = rng.choice(range(x.shape[0]), num, replace=False)
    return [x[choices] for x in (x, y)]


# ------------------------------------------
#  Training
# ------------------------------------------


def data_loader(x, y, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(x.shape[0])
    return ds.batch(batch_size)


def scheduler(decay_steps, decay_rate=0.5, lr=1e-3):
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate)


def optimizer(lr):
    return tf.optimizers.Adam(learning_rate=lr)


def backprop(model, loss, tape):
    trainable_vars = model.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    return zip(gradients, trainable_vars)


def train(model, x, y,
          batch_size,
          epochs,
          step_fn,
          optimizer_fn=optimizer,
          scheduler_fn=scheduler,
          verbose=1,
          verbose_every=1000):
    steps_per_epoch = int(np.ceil(x.shape[0] / batch_size))
    steps = epochs * steps_per_epoch

    scheduler = scheduler_fn(steps)
    optimizer = optimizer_fn(scheduler)

    loss_tracker = tf.keras.metrics.Mean(name='loss')
    mse_tracker = tf.keras.metrics.MeanSquaredError(name='mse')

    loader = data_loader(x, y, batch_size=batch_size)

    for epoch in range(1, epochs + 1):
        for x_batch, y_batch in loader:
            loss, y_pred = step_fn(model, optimizer, x_batch, y_batch)

            loss_tracker.update_state(loss)
            mse_tracker.update_state(y_batch, y_pred)

        if verbose and epoch % verbose_every == 0:
            print(f'epoch {epoch}: loss = {loss_tracker.result():.3f}, mse = {mse_tracker.result():.3f}')
            loss_tracker.reset_states()
            mse_tracker.reset_states()


# ------------------------------------------
#  Visualization
# ------------------------------------------


style = {
    'bg_line': {'ls': '--', 'c': 'black', 'lw': 1.0, 'alpha': 0.5},
    'fg_data': {'marker': '.', 'c': 'red', 'lw': 1.0, 'alpha': 1.0},
    'bg_data': {'marker': '.', 'c': 'gray', 'lw': 0.2, 'alpha': 0.2},
    'pred_sample': {'marker': 'x', 'c': 'blue', 'lw': 0.6, 'alpha': 0.5},
    'pred_mean': {'ls': '-', 'c': 'blue', 'lw': 1.0},
    'a_unc': {'color': 'lightgreen'},
    'e_unc': {'color': 'orange'},
}


def plot_data(x_train, y_train, x=None, y=None):
    if x is not None and y is not None:
        plt.plot(x, y, **style['bg_line'], label='f')
    plt.scatter(x_train, y_train, **style['fg_data'], label='Train data')
    plt.xlabel('x')
    plt.ylabel('y')


def plot_prediction(x, y_mean, y_samples=None, aleatoric_uncertainty=None, epistemic_uncertainty=None):
    x, y_mean, y_samples, epistemic_uncertainty, aleatoric_uncertainty = \
        flatten(x, y_mean, y_samples, epistemic_uncertainty, aleatoric_uncertainty)

    plt.plot(x, y_mean, **style['pred_mean'], label='Expected output')

    if y_samples is not None:
        plt.scatter(x, y_samples, **style['pred_sample'], label='Predictive samples')

    if aleatoric_uncertainty is not None:
        plt.fill_between(x,
                         y_mean + 2 * aleatoric_uncertainty,
                         y_mean - 2 * aleatoric_uncertainty,
                         **style['a_unc'], alpha=0.3, label='Aleatoric uncertainty')

    if epistemic_uncertainty is not None:
        plt.fill_between(x,
                         y_mean + 2 * epistemic_uncertainty,
                         y_mean - 2 * epistemic_uncertainty,
                         **style['e_unc'], alpha=0.3, label='Epistemic uncertainty')


def plot_uncertainty(x, aleatoric_uncertainty, epistemic_uncertainty=None):
    plt.plot(x, aleatoric_uncertainty, **style['a_unc'], label='Aleatoric uncertainty')

    if epistemic_uncertainty is not None:
        plt.plot(x, epistemic_uncertainty, **style['e_unc'], label='Epistemic uncertainty')

    plt.xlabel('x')
    plt.ylabel('Uncertainty')


def flatten(*ts):
    def _flatten(t):
        if t is not None:
            return tf.reshape(t, -1)

    return [_flatten(t) for t in ts]
