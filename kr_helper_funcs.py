"""
kr_helper_funcs.py - generic helper functions that can be used across Tensorflow & Keras DL code

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D

Usage:
    > Copy this file to any folder in your sys.path or leave in project folder
    > In the imports section, import this module as 
        import ke_helper_funcs as kru
"""

# imports & tweaks
import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

import tensorflow as tf
from tensorflow.keras import backend as K

USING_TF2 = (tf.__version__.startswith('2'))
seed = 123

os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)

if USING_TF2:
    tf.random.set_seed(seed)
else:
    tf.compat.v1.set_random_seed(seed)

# -----------------------------------------------------------------------------------------------
# Generic helper functions
# -----------------------------------------------------------------------------------------------


def progbar_msg(curr_tick, max_tick, head_msg, tail_msg, final=False):
    # --------------------------------------------------------------------
    # Keras like progress bar, used when copying files to show progress
    # --------------------------------------------------------------------
    progbar_len = 30
    len_max_tick = len(str(max_tick))

    if not final:
        prog = (curr_tick * progbar_len) // max_tick
        bal = progbar_len - (prog + 1)
        prog_msg = '  %s (%*d/%*d) [%s%s%s] %s%s' % (
            head_msg, len_max_tick, curr_tick, len_max_tick, max_tick, '=' * prog, '>', '.' * bal,
            tail_msg, ' ' * 35)
        print('\r%s' % prog_msg, end='', flush=True)
    else:
        prog_msg = '  %s (%*d/%*d) [%s] %s%s\n' % (
            head_msg, len_max_tick, max_tick, len_max_tick, max_tick, '=' * progbar_len, tail_msg,
            ' ' * 35)
        print('\r%s' % prog_msg, end='', flush=True)


def show_plots(history, metric=None, plot_title=None, fig_size=None):

    import seaborn as sns

    """ Useful function to view plot of loss values & 'metric' across the various epochs
        Works with the history object returned by the fit() or fit_generator() call """
    assert type(history) is dict

    # we must have at least loss in the history object
    assert 'loss' in history.keys(), f"ERROR: expecting \'loss\' as one of the metrics in history object"
    if metric is not None:
        assert isinstance(metric, str), "ERROR: expecting a string value for the \'metric\' parameter"
        assert metric in history.keys(), f"{metric} is not tracked in training history!"

    loss_metrics = ['loss']
    if 'val_loss' in history.keys():
        loss_metrics.append('val_loss')
    # after above lines, loss_metrics = ['loss', 'val_loss']

    other_metrics = []
    if metric is not None:
        other_metrics.append(metric)
        if f"val_{metric}" in history.keys():
            other_metrics.append(f"val_{metric}")
    # if metric is not None (e.g. if metrics = 'accuracy'), then other_metrics = ['accuracy', 'val_accuracy']

    # display the plots
    col_count = 1 if len(other_metrics) == 0 else 2
    df = pd.DataFrame(history)

    with sns.axes_style("darkgrid"):
        sns.set_context("notebook", font_scale=1.1)
        sns.set_style({"font.sans-serif": ["Verdana", "Arial", "Calibri", "DejaVu Sans"]})

        f, ax = plt.subplots(nrows=1, ncols=col_count, figsize=((16, 5) if fig_size is None else fig_size))
        axs = ax[0] if col_count == 2 else ax

        # plot the losses
        losses_df = df.loc[:, loss_metrics]
        losses_df.plot(ax=axs)
        #ax[0].set_ylim(0.0, 1.0)
        axs.grid(True)
        losses_title = 'Training \'loss\' vs Epochs' if len(
            loss_metrics) == 1 else 'Training & Validation \'loss\' vs Epochs'
        axs.title.set_text(losses_title)

        # plot the metric, if specified
        if metric is not None:
            metrics_df = df.loc[:, other_metrics]
            metrics_df.plot(ax=ax[1])
            ax[1].set_ylim(0.0, 1.0)
            ax[1].grid(True)
            metrics_title = f'Training \'{other_metrics[0]}\' vs Epochs' if len(other_metrics) == 1 \
                else f'Training & Validation \'{other_metrics[0]}\' vs Epochs'
            ax[1].title.set_text(metrics_title)

        if plot_title is not None:
            plt.suptitle(plot_title)

        plt.show()
        plt.close()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def time_taken_as_str(start_time, end_time):
    secs_elapsed = end_time - start_time

    SECS_PER_MIN = 60
    SECS_PER_HR = 60 * SECS_PER_MIN

    hrs_elapsed, secs_elapsed = divmod(secs_elapsed, SECS_PER_HR)
    mins_elapsed, secs_elapsed = divmod(secs_elapsed, SECS_PER_MIN)

    if hrs_elapsed > 0:
        ret = '%d hrs %d mins %d secs' % (hrs_elapsed, mins_elapsed, secs_elapsed)
    elif mins_elapsed > 0:
        ret = 'Time taken: %d mins %d secs' % (mins_elapsed, secs_elapsed)
    elif secs_elapsed > 1:
        ret = 'Time taken: %d secs' % (secs_elapsed)
    else:
        ret = 'Time taken - less than 1 sec'
    return ret


def save_model(model, base_file_name, save_dir=os.path.join('.', 'model_states')):
    """ save everything to one HDF5 file """

    # save the model
    if not base_file_name.lower().endswith('.h5'):
        base_file_name = base_file_name + '.h5'

    # base_file_name could be just a file name or complete path
    if (len(os.path.dirname(base_file_name)) == 0):
        # only file name specified e.g. kr_model.h5. We'll use save_dir to save
        if not os.path.exists(save_dir):
            # check if save_dir exists, else create it
            try:
                os.mkdir(save_dir)
            except OSError as err:
                print("Unable to create folder {} to save Keras model. Can't continue!".format(save_dir))
                raise err
        model_save_path = os.path.join(save_dir, base_file_name)
    else:
        # user passed in complete path e.g. './save_states/kr_model.h5'
        # create the directories if they don't exist
        save_dir = os.path.dirname(base_file_name)
        if not os.path.exists(save_dir):
            try:
                os.mkdir(save_dir)
            except OSError as err:
                print("Unable to create folder {} to save Keras model. Can't continue!".format(save_dir))
                raise err

        model_save_path = base_file_name

    #model_save_path = os.path.join(save_dir, base_file_name)
    model.save(model_save_path)
    print('Saved model to file %s' % model_save_path)


def load_model(base_file_name, save_dir=os.path.join('.', 'model_states'),
               custom_metrics_map=None, use_tf_keras_impl=True):
    """load model from HDF5 file"""
    if not base_file_name.lower().endswith('.h5'):
        base_file_name = base_file_name + '.h5'

    # base_file_name could be just a file name or complete path
    if (len(os.path.dirname(base_file_name)) == 0):
        # only file name specified e.g. kr_model.h5
        model_save_path = os.path.join(save_dir, base_file_name)
    else:
        # user passed in complete path e.g. './save_states/kr_model.h5'
        model_save_path = base_file_name

    if not os.path.exists(model_save_path):
        raise IOError('Cannot find model state file at %s!' % model_save_path)

    # load the state/weights etc.
    if use_tf_keras_impl:
        from tensorflow.keras.models import load_model
    else:
        from keras.models import load_model

    # load the state/weights etc. from .h5 file
    # @see: https://github.com/keras-team/keras/issues/3911
    # useful when you have custom metrics
    model = load_model(model_save_path, custom_objects=custom_metrics_map)
    print('Loaded Keras model from %s' % model_save_path)
    return model


def save_model_json(model, base_file_name, save_dir=os.path.join('.', 'model_states')):
    """ save the model structure to JSON & weights to HD5 """
    # check if save_dir exists, else create it
    if not os.path.exists(save_dir):
        try:
            os.mkdir(save_dir)
        except OSError as err:
            print("Unable to create folder {} to save Keras model. Can't continue!".format(save_dir))
            raise err

    # model structure is saved to $(save_dir)/base_file_name.json
    # weights are saved to $(save_dir)/base_file_name.h5
    model_json = model.to_json()
    json_file_path = os.path.join(save_dir, (base_file_name + ".json"))
    h5_file_path = os.path.join(save_dir, (base_file_name + ".h5"))

    with open(json_file_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5\n",
    model.save_weights(h5_file_path)
    print("Saved model to files %s and %s" % (json_file_path, h5_file_path))


def load_model_json(base_file_name, load_dir=os.path.join('.', 'keras_models'),
                    use_tf_keras_impl=True):
    """ loads model structure & weights from previously saved state """
    # model structure is loaded $(load_dir)/base_file_name.json
    # weights are loaded from $(load_dir)/base_file_name.h5

    if use_tf_keras_impl:
        from tensorflow.keras.models import model_from_json
    else:
        from keras.models import model_from_json

    # load model from save_path
    loaded_model = None
    json_file_path = os.path.join(load_dir, (base_file_name + ".json"))
    h5_file_path = os.path.join(load_dir, (base_file_name + ".h5"))

    if os.path.exists(json_file_path) and os.path.exists(h5_file_path):
        with open(json_file_path, "r") as json_file:
            loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(h5_file_path)
        print("Loaded model from files %s and %s" % (json_file_path, h5_file_path))
    else:
        msg = "Model file(s) not found in %s! Expecting to find %s and %s in this directory." % (
            load_dir, (base_file_name + ".json"), (base_file_name + ".h5"))
        raise IOError(msg)
    return loaded_model


def extract_files(arch_path, to_dir='.'):
    """extracts all files from a archive file (zip, tar. tar.bz2 file)
       at arch_path to the 'to_dir' directory """
    import os
    import tarfile
    import zipfile

    if os.path.exists(arch_path):
        supported_extensions = ['.zip', '.tar.gz', '.tgz', '.tar.bz2', '.tbz', '.npz']
        arch_exts = [arch_path.endswith(ext) for ext in supported_extensions]

        if np.any(arch_exts):
            # if extension is any of our supported extension, we are ok
            opener_triplets = [('zipfile.ZipFile', zipfile.ZipFile, 'r'),
                               ('tarfile.open (for .tar.gz or .tgz file)', tarfile.open, 'r:gz'),
                               ('tarfile.open (for .tar.bz2 or .tbz file)', tarfile.open, 'r:bz2')]
            opened_successfully = False
            curr_dir = os.getcwd()
            os.chdir(to_dir)

            try:
                for opener_str, opener, mode in opener_triplets:
                    try:
                        # try various options to open archive file
                        with opener(arch_path, mode) as f:
                            opened_successfully = True
                            print(f"Extracting files from archive using {opener_str} opener...", flush=True)
                            f.extractall()
                            break
                    except:
                        continue
            finally:
                os.chdir(curr_dir)
                if not opened_successfully:
                    raise ValueError(f"Could not extract '{arch_path}' as no appropriate extractor is found")
        else:
            raise ValueError(f"Unsupported archive file {arch_path} - only one of {supported_extensions} supported")
    else:
        raise ValueError(f"{arch_path} - path does not exist!")

# ----------------------------------------------------------------------------------------
# custom metrics that can be tracked
# ----------------------------------------------------------------------------------------


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))


def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))
