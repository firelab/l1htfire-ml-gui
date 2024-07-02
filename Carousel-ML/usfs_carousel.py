"""Demonstrates the use of a carousel model on carousel data.

NOTE: COPY/PASTED CODE from carousel_v3.py for USFS collaborators.

Setup:

# For tkinter, which matplotlib uses to make plots.
sudo apt-get install python3-tk

# For the modules used in this python code.
pip install numpy
pip install pandas
pip install ml_collections
pip install matplotlib

To use this:

1) Update the value in get_io_config for the cfg.io_path to be the path where
   all the data and model files are stored that you want to use.

2) Set the cfg.f_mode and cfg.l_mode in get_data_config to indicate which model
   and files you're loading.

   For the model and files using features with units and labels that are not
   logged:

     cfg.f_mode = 'unit'
     cfg.l_mode = 'norm'

   For the model and files using features without units and log labels:

     cfg.f_mode = 'unitless'
     cfg.l_mode = 'log'

   Note, it's up to you to make sure you set the io_path to a directory that
   contains the files that are used for the setting the f_mode and l_mode that's
   appropriate for that directory.

The command to run:

python3 your/path/to/usfs_carousel.py

Misc Notes:

* The script uses a ConfigDict as a way to configure everything.  Think of a
  ConfigDict as a nested dictionary that allows nice syntax for accessing=
  fields.

* It'll be easiest to understand this code by reading from the bottom-up.  The
  function run() basically shows how to do everything.

* The function get_unnorm_pred_batch is the function that actually uses the
  model to make predictions.  It takes in a dataframe of unnormalized data, then
  it makes a copy, and then normalizes the results, then makes a prediction,
  then unnormalizses the prediction.  This is the only place in the code you
  have to care about normalization details.
"""

from absl import flags
from typing import Any, Dict, List, Optional, Tuple

import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as co
matplotlib.use('TkAgg')
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf

from ml_collections import config_dict as cd

STATIC_FEATURES_NORMALIZED = np.array([[-1.43293148,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        , -1.01436574, -0.33490194,  0.8566    ,  0.48074986],
       [-1.05480562,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        , -0.74669214, -0.2465271 ,  0.8566    ,  0.48074986],
       [-0.83361617,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        , -0.59011313, -0.19483114,  0.8566    ,  0.48074986],
       [-0.67667976,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        , -0.47901855, -0.15815227,  0.8566    ,  0.48074986],
       [-0.55495042,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        , -0.3928469 , -0.12970193,  0.8566    ,  0.48074986],
       [-0.45549031,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        , -0.32243954, -0.1064563 ,  0.8566    ,  0.48074986],
       [-0.37139798,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        , -0.26291096, -0.08680241,  0.8566    ,  0.48074986],
       [-0.2985539 ,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        , -0.21134496, -0.06977743,  0.8566    ,  0.48074986],
       [-0.23430086,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        , -0.16586052, -0.05476034,  0.8566    ,  0.48074986],
       [-0.17682456,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        , -0.12517331, -0.04132709,  0.8566    ,  0.48074986],
       [-0.12483092,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        , -0.08836725, -0.02917524,  0.8566    ,  0.48074986],
       [-0.07736445,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        , -0.05476595, -0.01808147,  0.8566    ,  0.48074986],
       [-0.03369953,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        , -0.02385574, -0.00787619,  0.8566    ,  0.48074986],
       [ 0.00672788,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.00476263,  0.00157243,  0.8566    ,  0.48074986],
       [ 0.04436489,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.0314057 ,  0.01036887,  0.8566    ,  0.48074986],
       [ 0.07957196,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.05632863,  0.0185974 ,  0.8566    ,  0.48074986],
       [ 0.11264392,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.07974013,  0.02632692,  0.8566    ,  0.48074986],
       [ 0.143825  ,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.10181307,  0.0336145 ,  0.8566    ,  0.48074986],
       [ 0.17331977,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.12269228,  0.04050796,  0.8566    ,  0.48074986],
       [ 0.2013013 ,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.14250028,  0.04704775,  0.8566    ,  0.48074986],
       [ 0.22791732,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.16134165,  0.05326839,  0.8566    ,  0.48074986],
       [ 0.25329494,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.17930634,  0.0591996 ,  0.8566    ,  0.48074986],
       [ 0.27754428,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.19647234,  0.0648671 ,  0.8566    ,  0.48074986],
       [ 0.30076141,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.21290765,  0.07029337,  0.8566    ,  0.48074986],
       [ 0.32303064,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.22867193,  0.07549809,  0.8566    ,  0.48074986],
       [ 0.34442633,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.24381785,  0.08049865,  0.8566    ,  0.48074986],
       [ 0.36501445,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.25839208,  0.08531046,  0.8566    ,  0.48074986],
       [ 0.38485374,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.27243623,  0.08994726,  0.8566    ,  0.48074986],
       [ 0.40399676,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.28598749,  0.09442133,  0.8566    ,  0.48074986],
       [ 0.42249075,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.2990793 ,  0.09874371,  0.8566    ,  0.48074986],
       [ 0.44037826,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.31174179,  0.10292434,  0.8566    ,  0.48074986],
       [ 0.45769782,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.32400223,  0.10697224,  0.8566    ,  0.48074986],
       [ 0.47448439,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.33588536,  0.11089556,  0.8566    ,  0.48074986],
       [ 0.49076978,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.34741372,  0.11470175,  0.8566    ,  0.48074986],
       [ 0.50658307,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.35860787,  0.1183976 ,  0.8566    ,  0.48074986],
       [ 0.52195086,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.36948666,  0.12198933,  0.8566    ,  0.48074986],
       [ 0.53689756,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.38006736,  0.12548265,  0.8566    ,  0.48074986],
       [ 0.55144563,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.39036587,  0.12888279,  0.8566    ,  0.48074986],
       [ 0.56561578,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.40039686,  0.13219461,  0.8566    ,  0.48074986],
       [ 0.57942716,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.41017388,  0.13542258,  0.8566    ,  0.48074986],
       [ 0.59289748,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.41970945,  0.13857084,  0.8566    ,  0.48074986],
       [ 0.60604319,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.42901524,  0.14164323,  0.8566    ,  0.48074986],
       [ 0.61887955,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.43810204,  0.14464332,  0.8566    ,  0.48074986],
       [ 0.63553795,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.44989445,  0.14853669,  0.8566    ,  0.48074986],
       [ 0.6517027 ,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.4613374 ,  0.15231468,  0.8566    ,  0.48074986],
       [ 0.66740221,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.47245102,  0.15598394,  0.8566    ,  0.48074986],
       [ 0.68266251,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.48325372,  0.15955055,  0.8566    ,  0.48074986],
       [ 0.69750752,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.49376243,  0.16302009,  0.8566    ,  0.48074986],
       [ 0.71195923,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.50399273,  0.16639772,  0.8566    ,  0.48074986],
       [ 0.72603796,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.513959  ,  0.16968817,  0.8566    ,  0.48074986],
       [ 0.73976246,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.52367451,  0.17289583,  0.8566    ,  0.48074986],
       [ 0.75315014,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.53315159,  0.17602477,  0.8566    ,  0.48074986],
       [ 0.76621712,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.54240164,  0.17907876,  0.8566    ,  0.48074986],
       [ 0.77897841,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.5514353 ,  0.18206131,  0.8566    ,  0.48074986],
       [ 0.79144799,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.56026246,  0.18497567,  0.8566    ,  0.48074986],
       [ 0.80363889,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.56889235,  0.18782491,  0.8566    ,  0.48074986],
       [ 0.81556331,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.57733359,  0.19061186,  0.8566    ,  0.48074986],
       [ 0.82723263,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.58559425,  0.19333919,  0.8566    ,  0.48074986],
       [ 0.83865756,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.59368191,  0.1960094 ,  0.8566    ,  0.48074986],
       [ 0.86081371,  0.23077204,  0.96369933,  0.        ,  0.61697568,
         0.        ,  0.60936615,  0.2011877 ,  0.8566    ,  0.48074986]])

DYNAMIC_FEATURES_NORMALIZED = np.array([[-18.82064282],
       [-15.71435911],
       [-13.89729961],
       [-12.60807539],
       [-11.60807539],
       [-10.79101589],
       [-10.10020194],
       [ -9.50179167],
       [ -8.9739564 ],
       [ -8.50179167],
       [ -8.07466671],
       [ -7.68473217],
       [ -7.32602717],
       [ -6.99391822],
       [ -6.68473217],
       [ -6.39550795],
       [ -6.12382355],
       [ -5.86767268],
       [ -5.62537475],
       [ -5.39550795],
       [ -5.17685872],
       [ -4.96838299],
       [ -4.76917597],
       [ -4.57844845],
       [ -4.39550795],
       [ -4.21974345],
       [ -4.05061319],
       [ -3.8876345 ],
       [ -3.73037555],
       [ -3.57844845],
       [ -3.43150348],
       [ -3.28922423],
       [ -3.1513235 ],
       [ -3.01753983],
       [ -2.8876345 ],
       [ -2.76138896],
       [ -2.63860265],
       [ -2.51909103],
       [ -2.40268396],
       [ -2.28922423],
       [ -2.17856626],
       [ -2.070575  ],
       [ -1.96512496],
       [ -1.82827707],
       [ -1.69548454],
       [ -1.56651391],
       [ -1.44115132],
       [ -1.31920039],
       [ -1.20048033],
       [ -1.08482436],
       [ -0.97207829],
       [ -0.86209927],
       [ -0.75475473],
       [ -0.64992141],
       [ -0.54748449],
       [ -0.44733686],
       [ -0.34937842],
       [ -0.25351551],
       [ -0.15966035],
       [  0.02235127]])

def get_io_config() -> cd.ConfigDict:
  """Returns the IO config."""
  cfg = cd.ConfigDict()

  # The path to the raw data csv that is the exported file given to me by Isaac.
  cfg.input_raw_csv_path = r"C:\Users\Natha\Documents\Carousel\John's Carousel App\carousel (1).csv"

  # The ath that all files should be saved/read to/from.
  cfg.io_path = (
      r"C:\Users\Natha\Documents\Carousel\John's Carousel App\unitless_log")

  # The filename for the model.
  cfg.output_model_filename = 'model.h5'

  # The filename to save the full dataset to.
  cfg.output_data_filename = 'full_data.csv'

  # The filename to save the training, testing and validation data to.
  cfg.output_train_filename = 'train_data.csv'
  cfg.output_test_filename = 'test_data.csv'
  cfg.output_val_filename = 'val_data.csv'

  # The name of the file to save the normalization coefficients to.
  cfg.output_norm_filename = 'norm_coeffs.json'

  return cfg


def get_data_config() -> cd.ConfigDict:
  """Returns the data config."""
  cfg = cd.ConfigDict()

  # The columns in the input data that are the static columns.
  cfg.static_features = ()

  # The column in the input data that is the dynamic one.
  # FILLED IN AUTOMATICALLY.
  cfg.dynamic_feature = ''

  # The column in the input data that is the dynamic label.
  # FILLED IN AUTOMATICALLY.
  cfg.dynamic_label = ''

  # The meta column that identifies which rows get grouped together to make the
  # same data point.
  cfg.group_column = 'burn'

  # Which set of features to use:  'unit', 'unitless'.
  cfg.f_mode = 'unitless'

  # Which set of labels to use: 'norm', 'log'.
  cfg.l_mode = 'log'

  return cfg


def get_plot_config() -> cd.ConfigDict:
  """Returns the plot config."""
  cfg = cd.ConfigDict()

  cfg.sweep_cmap = 'twilight'

  return cfg


def get_config() -> cd.ConfigDict:
  """Returns the entire config."""
  cfg = cd.ConfigDict()

  cfg.io = get_io_config()
  cfg.data = get_data_config()
  cfg.plot = get_plot_config()

  return cfg


def get_unnorm_pred_batch(
    cfg: cd.ConfigDict,
    df: pd.DataFrame,
    model: tf.keras.Model,
    norm_coeffs: Dict[str, Tuple[float, float]],
) -> pd.Series:
  """Performs batch predictions on unnorm'd data returning unnorm'd results."""
  df_copy = df.copy()

  # Extract and Normalize Static Features
  static_features_normalized = (
      df_copy[list(cfg.data.static_features)].apply(
          lambda col: (
              (col - norm_coeffs[col.name][0]) /
              (norm_coeffs[col.name][1] - norm_coeffs[col.name][0])),
          axis=0,
      )).values

  # Extract and Normalize Dynamic Feature
  dynamic_feature = df_copy[[cfg.data.dynamic_feature]].values
  dynamic_feature_normalized = (
      (dynamic_feature - norm_coeffs[cfg.data.dynamic_feature][0]) / (
          norm_coeffs[cfg.data.dynamic_feature][1]
          - norm_coeffs[cfg.data.dynamic_feature][0]))

  # Make Predictions
  predictions_normalized = model.predict(
      [static_features_normalized, dynamic_feature_normalized]
  )

  # Denormalize Predictions
  predictions_denormalized = (
      predictions_normalized
      * (
          norm_coeffs[cfg.data.dynamic_label][1]
          - norm_coeffs[cfg.data.dynamic_label][0]
      )
      + norm_coeffs[cfg.data.dynamic_label][0]
  )
  return predictions_denormalized.flatten()


def plot_pred(
    cfg: cd.ConfigDict,
    df: pd.DataFrame,
    model: tf.keras.Model,
    norm_coeffs: Dict[str, Tuple[float, float]],
    axes,
    plot_labels: bool = True,
    label: str = 'pred',
    color: str = 'black'):
  """Plots the prediction of a single burn."""
  # Perform the prediction.
  pred = get_unnorm_pred_batch(cfg, df, model, norm_coeffs)

  # Plot the prediction.
  x_values = df[cfg.data.dynamic_feature].values
  y_values = df[cfg.data.dynamic_label].values

  # Plot the prediction.
  axes.plot(x_values, pred, '-', marker='', color=color, label=label)

  # Plot the labels.
  if plot_labels:
    axes.plot(x_values, y_values, linestyle='none', marker='.',
              color='red', label='label')
  axes.legend()


def cycle_through_preds(
    cfg: cd.ConfigDict,
    df: pd.DataFrame,
    model: tf.keras.Model,
    norm_coeffs: Dict[str, Tuple[float, float]],
    num_rows_output: int = 2,
    num_cols_output: int = 2):
  """Randomly picks preds and evals them.  Ctrl-C to stop."""
  burns = df[cfg.data.group_column].unique()
  _, axes = plt.subplots(
      num_rows_output, num_cols_output, figsize=(10, 10), squeeze=False)
  for i in range(num_rows_output):
    for j in range(num_cols_output):
      rand_index = random.randint(0, len(burns)-1)
      rand_burn = burns[rand_index]
      print('Showing results for: ', rand_burn)
      target_rows = df[df[cfg.data.group_column] == rand_burn]
      plot_pred(cfg, target_rows, model, norm_coeffs, axes[i][j])
      axes[i][j].set_title(rand_burn)
  plt.show()


def get_shades(
    cfg: cd.ConfigDict, min_val: float, max_val: float, step: float) -> (
        list[str]):
  """Returns a sequence of shades."""
  values = np.arange(min_val, max_val + step, step)
  cmap = plt.get_cmap(cfg.plot.sweep_cmap)

  spread = values.max() - values.min()
  vmin = values.min() - 0.5 * spread
  norm = plt.Normalize(vmin=vmin, vmax=values.max())

  colors = cmap(norm(values))

  hex_colors = [co.to_hex(c) for c in colors]
  return hex_colors


def pred_sweep(
    cfg: cd.ConfigDict,
    model: tf.keras.Model,
    df: pd.DataFrame,
    norm_coeffs: Dict[str, Tuple[float, float]],
    target: str,
    sweep_feature: str,
    sweep_range: Tuple[float, float, float]):
  """Performs feature sweep.

  Args:
    cfg: The config.
    model: The model to do the predictions on.
    df: The data containing the rows.
    norm_coeffs:  The normalization coefficients.
    target: Which set of data points to use in df.
    sweep_feature: Which feature to sweep across.
    sweep_range: The range of value to sweep across.  (min, max, step).
  """
  fig, axes = plt.subplots(1, 2, figsize=(10, 10), squeeze=False)

  # Cycle through the parameter sweep, adding the curve for each new value.
  target_rows = df[df[cfg.data.group_column] == target].copy()
  colors = get_shades(cfg, *sweep_range)
  for sweep_value, color in zip(np.arange(*sweep_range), colors):
    target_rows[sweep_feature] = sweep_value
    plot_pred(cfg, target_rows, model, norm_coeffs, axes[0][1],
              plot_labels=False,
              color=color,
              label=f'{sweep_value}')

  # Show the ground truth and the prediction for the given burn.
  target_rows = df[df[cfg.data.group_column] == target].copy()
  plot_pred(
      cfg, target_rows, model, norm_coeffs, axes[0][0],
      label=f'pred: {sweep_feature} = {target_rows[sweep_feature].iloc[0]}')

  axes[0][1].set_xlim(0.0)
  axes[0][1].set_ylim(0.0)
  axes[0][0].set_xlim(axes[0][1].get_xlim())
  axes[0][0].set_ylim(axes[0][1].get_ylim())
  fig.suptitle(f'burn={target} sweep on {sweep_feature}')

  # Show the plot.
  plt.show()


def setup_data_columns(cfg: cd.ConfigDict):
  """Sets up the config for various experiments.

  Args:
    cfg: the config.  It will be mutated.
    f_mode: determines which columns to use for features.
      'unit': Use unit features with for static features.
      'unitless': Use the unitless static features.
    l_mode: determines which columns to use for labels.
      'norm': Use the non-logged values.
      'log':  Use the log values.
  """
  if cfg.data.f_mode == 'unit':
    print('Setting feature config to use unit features.')
    cfg.data.static_features = (
        'Angle.deg',
        'FL(m)',
        'FlameDepth.m',
        'FlameVel.mps',
        'Intensity.kwm',
        'WindSpd.mps',
        'slope.deg',
        'width.meters',
    )
  elif cfg.data.f_mode == 'unitless':
    print('Setting feature config to use unitless features.')
    cfg.data.static_features = (
        'd*',
        'a*',
        'f*',
        's*',
        'w*',
        'd*s*',
        'd*w*',
        'd*a*',
        'r*',
        'Z*',
    )
  else:
    raise ValueError('Unknown f_mode: ', f_mode)

  if cfg.data.l_mode == 'norm':
    print('Setting up config to use norm labels.')
    cfg.data.dynamic_feature = 'Distance.m'
    cfg.data.dynamic_label = 'Temp.C'
  elif cfg.data.l_mode == 'log':
    print('Setting up config to use log labels.')
    cfg.data.dynamic_feature = 'ln(dist)'
    cfg.data.dynamic_label = 'ln(Temp.C)'
  else:
    raise ValueError('Unknown l_mode: ', l_mode)


def load_all(cfg: cd.ConfigDict) -> (
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
          tf.keras.Model, Dict[str, Tuple[float, float]]]):
  """Loads all files specified in the configuration."""
  setup_data_columns(cfg)
  io_path = cfg.io.io_path
  if not io_path:
    raise ValueError('No io_path: ', cfg.io.io_path)

  raw_data = pd.DataFrame()
  model = None
  train_data = pd.DataFrame()
  test_data = pd.DataFrame()
  val_data = pd.DataFrame()
  norm_coeffs = {}

  # Load raw data
  if cfg.io.input_raw_csv_path:
    print('Loading full data.')
    raw_data = pd.read_csv(os.path.join(io_path, cfg.io.input_raw_csv_path))

  # Load model
  if cfg.io.output_model_filename:
    print('Loading model.')
    model_path = os.path.join(io_path, cfg.io.output_model_filename)
    model = tf.keras.models.load_model(model_path)

  # Load full dataset
  if cfg.io.output_data_filename:
    print('Loading output data.')
    raw_data = pd.read_csv(
        os.path.join(io_path, cfg.io.output_data_filename))

  # Load train, test, and validation data
  if cfg.io.output_train_filename:
    print('Loading training data.')
    train_data = pd.read_csv(
        os.path.join(io_path, cfg.io.output_train_filename))
  if cfg.io.output_test_filename:
    print('Loading testing data.')
    test_data = pd.read_csv(
        os.path.join(io_path, cfg.io.output_test_filename))
  if cfg.io.output_val_filename:
    print('Loading validation data.')
    val_data = pd.read_csv(
        os.path.join(io_path, cfg.io.output_val_filename))

  # Load normalization coefficients
  if cfg.io.output_norm_filename:
    print('Loading normalization coefficients.')
    norm_path = os.path.join(io_path, cfg.io.output_norm_filename)
    with open(norm_path, 'r') as f:
      norm_coeffs = json.load(f)

  return raw_data, train_data, val_data, test_data, model, norm_coeffs


def run():
  """Loads the data and models and plots some predictions."""
  cfg = get_config()
  print('The config was:\n', cfg, end='\n')

  # Load in all the data.
  raw_data, train_data, val_data, test_data, model, norm_coeffs = load_all(cfg)

  # The model was:
  model.summary()

  # Looking at a few predictions on test data.
  #cycle_through_preds(cfg, test_data, model, norm_coeffs, 3, 3)

  # Do a parameter sweep on the model.
  test_burns = test_data[cfg.data.group_column].unique()
  pred_sweep(
      cfg, model, test_data, norm_coeffs, test_burns[0], 'Intensity.kwm',
      (0.0, 400.0, 20.0))
  
  return cfg, raw_data, train_data, val_data, test_data, model, norm_coeffs


if __name__ == '__main__':
  run()
