from typing import List
import json
import streamlit as st
import tensorflow as tf
import os.path
import pandas as pds
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from ml_collections import config_dict as cd


tf.compat.v1.enable_eager_execution()

st.set_page_config(page_title="Carousel ML", page_icon=":fire:", layout="wide")

class Static_Feature(object):
    def __init__(self, short_name, display_name, gui_min, min, max, default, step):
        self.short_name = short_name
        self.display_name = display_name
        self.gui_min = gui_min
        self.min = min
        self.max = max
        self.default = default
        self.step = step
        
    def get_min(self):
        return self.min

    def get_max(self):
        return self.max


#The order of this list dictates the order in which the features are passed to the ML model
STATIC_FEATURE_LOOKUP = [
	Static_Feature("Angle.deg", "Angle (degrees)", 0.0, 0.00001, 90.0, 20.0, 1.0),
	Static_Feature("FL(m)", "Flame Length (m)", 0.0, 0.00001, 1.0, 0.3, 0.05), # True max = .60687
	Static_Feature("FlameDepth.m", "Flame Depth (m)", 0.1, 0.1, 20.0, 0.4, .1),
	Static_Feature("FlameVel.mps", "Flame Velocity (m/s)", 0.1, 0.00001, 2.0, 0.4, 0.1), # True max =  1.72587
	Static_Feature("Intensity.kwm", "Intensity (kwm)", 40.0, 40.0, 1000.0, 85.0, 5.0),
	Static_Feature("WindSpd.mps", "Wind Speed (m/s)", 0.0, 0.00001, 6.0, 1.0, 0.1),
	Static_Feature("slope.deg", "Slope (degrees)", 0.0, 0.00001, 45.0, 5.0, 0.5),
	Static_Feature("width.meters", "Width (m)", 0.5, 0.5, 100.0, 2.6, 0.5),
]

#Modules borrowed from Google

def get_io_config() -> cd.ConfigDict:
  """Returns the IO config."""
  cfg = cd.ConfigDict()

#   # The path to the raw data csv that is the exported file given to me by Isaac.
# #   cfg.input_raw_csv_path = r"C:\Users\Natha\Documents\Carousel\John's Carousel App\carousel (1).csv"

# #   # The ath that all files should be saved/read to/from.
# #   cfg.io_path = (
# #       r"C:\Users\Natha\Documents\Carousel\John's Carousel App\unitless_log")

#   # The filename for the model.
#   cfg.output_model_filename = 'model.h5'

#   # The filename to save the full dataset to.
#   cfg.output_data_filename = 'full_data.csv'

#   # The filename to save the training, testing and validation data to.
#   cfg.output_train_filename = 'train_data.csv'
#   cfg.output_test_filename = 'test_data.csv'
#   cfg.output_val_filename = 'val_data.csv'

#   # The name of the file to save the normalization coefficients to.
#   cfg.output_norm_filename = 'norm_coeffs.json'

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
  cfg.f_mode = 'unit'

  # Which set of labels to use: 'norm', 'log'.
  cfg.l_mode = 'norm'

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
    df: pds.DataFrame,
    model: tf.keras.Model,
    norm_coeffs
) -> pds.Series:
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
    #print('Setting feature config to use unit features.')
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
    #print('Setting feature config to use unitless features.')
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
    #print('Setting up config to use norm labels.')
    cfg.data.dynamic_feature = 'Distance.m'
    cfg.data.dynamic_label = 'Temp.C'
  elif cfg.data.l_mode == 'log':
    #print('Setting up config to use log labels.')
    cfg.data.dynamic_feature = 'ln(dist)'
    cfg.data.dynamic_label = 'ln(Temp.C)'
  else:
    raise ValueError('Unknown l_mode: ', l_mode)

#end modules borrowed from Google

def load_model():
    # change model here if needed
    model_path = str(os.getcwd() + '/unit_norm/model.h5')
    model = tf.keras.models.load_model(model_path)
    return model 

def load_norm_coefficients():
	norm_coeffs = {}
	norm_path = str(os.getcwd() + '/unit_norm/norm_coeffs.json')
	with open(norm_path, 'r') as f:
		norm_coeffs = json.load(f)
	
	return norm_coeffs;

# for burn_lookup_mode
def select_cfg():
    #change unit & norm here if needed
    cfg = get_config()
    cfg.data.l_mode = "norm"
    cfg.data.f_mode = "unit"
    return cfg

def get_csv_data(cfg):
    data_path = str(os.getcwd() + '/unit_norm/test_data.csv')
    setup_data_columns(cfg)
    data = pds.read_csv(data_path)
    return data

def get_burn_names(data):
    burns = data["burn"].unique()
    burns = burns.tolist()
    return burns

def get_selected_burn_data(cfg, burn_selection):
    data = get_csv_data(cfg)
    data_1 = data[data["burn"]==burn_selection]
    
    true_values = data_1["Temp.C"].tolist()
    x_value_list = data_1["Distance.m"].tolist()
    ml_pred = get_unnorm_pred_batch(cfg, data_1, load_model(), load_norm_coefficients())
    stat_pred_list = []
    for idx in range(len(data_1)):
        stat_pred = calculate_ln_temp(x = data_1["Distance.m"].iloc[idx], intensity = data_1["Intensity.kwm"].iloc[idx], slope = data_1["slope.deg"].iloc[idx], angle = data_1["Angle.deg"].iloc[idx], windspeed= data_1["WindSpd.mps"].iloc[idx], fzdepth =  data_1["FlameDepth.m"].iloc[idx], fzwidth = data_1["width.meters"].iloc[idx])
        stat_pred_list.append(np.exp(stat_pred))
        
    return x_value_list, true_values, ml_pred, stat_pred_list

    
def calculate_error_for_test_burns(data, burn_selection, true_values, ml_pred, stat_pred_list):
    data_1 = data_1 = data[data["burn"]==burn_selection]
    ml_error_list = []
    stat_error_list = []
    for i in range(len(data_1)):
        ml_error = abs(ml_pred[i]-true_values[i])
        stat_error = abs(stat_pred_list[i]- true_values[i])
        ml_error_list.append(ml_error)
        stat_error_list.append(stat_error)
    avg_ml_error = np.mean(ml_error_list)
    avg_stat_error = np.mean(stat_error_list)
    return avg_ml_error, avg_stat_error

def generate_level_variable_input_valuesII(level_variable : Static_Feature, slider_values : dict):
    '''
    design new way to distribute level variables. Instead of a spread across all possible values, lets just do 5, where the match to the slider is the center, with two distributed off that plot
    I dont have to disable the sliders at that point, and our ml predictions will perfectly match. We will need which variable is level variable as well as the slider_value
    Inputs: 
    level_variable: which input is to be varied
    slider_values: what is the selected slider value for that variable
    '''
    level_variable_traces_count = 4
    
    for slider_names in slider_values.keys():
        if level_variable.short_name == slider_names:
            levelv_slider_value = slider_values[slider_names]
    
    level_variable_input_values = []
    
    level_variable_input_values.append(levelv_slider_value)
    
    axis_min = level_variable.min
    axis_max = level_variable.max
    
    total_range = axis_max - axis_min
    
    step = total_range / (level_variable_traces_count+2)
    
    val = axis_min + step
    
    while len(level_variable_input_values) < (level_variable_traces_count + 1):
        level_variable_input_values.append(val)
        val += step
    return level_variable_input_values
  

def get_x_axis_points(x_range_min, x_range_max, x_value_sample_count):
	'''
	Handles case of user selected domain
	
	Inputs: user selected min, max and number of points 
	Returns: list of x axis points
 	'''
	x_axis_points = []
	val = x_range_min
	step = (x_range_max-x_range_min)/x_value_sample_count
	while len(x_axis_points) < x_value_sample_count:
		x_axis_points.append(val) 
		val += step
	return x_axis_points

def combine_prediction_inputs(level_var_input_values, slider_values, level_variable: Static_Feature):
    '''
    Returns a list of dicts, where each dict is of length {number of static features}
    and the level variable that has been user-selected is replaced
    by the ten level_var_input_values obtained from generate_level_variable_input_values

    Inputs:
    level_var_input_values: list of 10 values for selected level_variable
    slider_values: list of user-selected values for static inputs
    level_variable: the static_feature that was selected by the user to be the level_variable

    Returns: list of dicts of slider values with different input variable values to be used to make predictions
    '''
    prediction_inputs = []
    for val in level_var_input_values:
        slider_values_copy = slider_values.copy()
        slider_values_copy[level_variable.short_name] = val
        slider_values_copy["FL(m)"] = calculate_fl_from_sliders(slider_values_copy["Intensity.kwm"], slider_values_copy["FlameDepth.m"])
        slider_values_copy["FlameVel.mps"] = calculate_flamevel_from_sliders(slider_values_copy["Intensity.kwm"])
        prediction_inputs.append(slider_values_copy)
    return prediction_inputs

def calculate_fl_from_sliders(intensity, fzdepth):
    return 0.0105*intensity**0.774/fzdepth**0.161

def calculate_flamevel_from_sliders(intensity):
    return ((2*9.81*intensity*1000)/(1.6141*1007*300))**(1/3)

def order_pred_input_dict(prediction_input_dict) -> list:
	'''
	converts dictionary of static values to a list in the correct order,
	(so long as STATIC_FEATURE_LOOKUP is in correct order) that can be converted into a tensor

	prediction_input_dict: dictionary of all static values for a given prediction

	Returns: list of static values in the order that the model needs them
	'''
	short_names = [feature.short_name for feature in STATIC_FEATURE_LOOKUP]

	pred_input_list = []
	for short_name in short_names:
		pred_input_list.append(prediction_input_dict[short_name])
	
	return pred_input_list

def normalize_static_features(static_feature_inputs: list, norm_coeffs: dict):
	'''normalizes each static_feature in the input list.
	obv heavily dependent upon:
	 1. using every feature in the lookup table
	 2. static_feature_inputs being in the same order as the STATIC_FEATURE_LOOKUP'''
	static_features_normalized = []

	for i in range(len(STATIC_FEATURE_LOOKUP)):
		name = STATIC_FEATURE_LOOKUP[i].short_name
		static_feature_value = static_feature_inputs[i]
		norm_coeff_min = norm_coeffs[name][0]
		norm_coeff_max = norm_coeffs[name][1]

		static_feature_normalized = (
			(static_feature_value - norm_coeff_min) /
			(norm_coeff_max - norm_coeff_min))
		
		static_features_normalized.append(static_feature_normalized)
	
	return static_features_normalized

def normalize_dynamic_feature(dynamic_feature, norm_coeffs):
	dynamic_feature_normalized = []

	for i in range(len(dynamic_feature)):
		value_normalized = (
			(dynamic_feature[i] - norm_coeffs["Distance.m"][0]) /
			(norm_coeffs["Distance.m"][1] - norm_coeffs["Distance.m"][0])
		)
		dynamic_feature_normalized.append(value_normalized)

	return dynamic_feature_normalized;

def denormalize_predictions(predictions_normalized, norm_coeffs):
	predictions_denormalized = (
		predictions_normalized
		* (norm_coeffs["Temp.C"][1] - norm_coeffs["Temp.C"][0])
		+ norm_coeffs["Temp.C"][0]
	)

	return predictions_denormalized;

def populate_static_array(static_list: list, x_points: list):
	mega_list = []
	for _ in range(len(x_points)):
		mega_list.append(static_list)
	return np.array(mega_list)

def populate_dynamic_array(lst: list):
	mega_list = []
	for x in lst:
		mega_list.append([x])
	return np.array(mega_list)

def create_pred_set(pred_input_list: list, x_points: list, norm_coeffs: dict):
	'''
	takes list of inputs and feeds to ml model to display e^(resultant y-values)
	old comment:
	takes list of inputs and feeds to ml model, returns list of predictions
	TODO: does the model produe a linear equation and not just output points??
	I'm taking the log of the list of x-coordinates and feeding them into the linear equation produced by the model
	The e^(resultant y-values) is what is displayed on the graph

	Inputs 
 	pred_input_list: list of slider values + one level variable entry already in the order they need to be in
	x_points: points on x axis to gather predictions for

	Returns: predictions denormalized
    '''

	static_features_normalized = normalize_static_features(pred_input_list, norm_coeffs)
	dynamic_feature_normalized = normalize_dynamic_feature(x_points, norm_coeffs)

	# return an array of shape len(x_points) x 8
	static_arr = populate_static_array(static_features_normalized, x_points)
	
    # return an array of shape len(x_points) x 1
	dynamic_arr = populate_dynamic_array(dynamic_feature_normalized)
 
	model = load_model()	
    # run prediction model with normalized data
	predictions_normalized = model.predict([static_arr, dynamic_arr])

	# denormalize outputs
	predictions_denormalized = denormalize_predictions(predictions_normalized, norm_coeffs)
	predictions_denormalized_flat = predictions_denormalized.flatten()
 
	return predictions_denormalized_flat;

def get_prediction_output(level_variable: Static_Feature, level_var_input_values: list, slider_values: dict, x_points: list):
	'''
	Inputs 
 	level_var_name: name of level variable to be looked up in x_index_dict
	slider_values: list of widget input slider values
	x_points: points to be predicted on
 
	Returns: list of y output predictions for stepped level variables 
  	'''
	
	prediction_inputs = combine_prediction_inputs(level_var_input_values, slider_values, level_variable) 
	norm_coeffs = load_norm_coefficients()

	y_set = []
	for prediction_input_dict in prediction_inputs:
		pred_input_list = order_pred_input_dict(prediction_input_dict)
		y_set.append(create_pred_set(pred_input_list, x_points, norm_coeffs))
	return y_set

def calculate_flamelength(intensity, fzdepth):
    return 0.0105 * ((intensity**0.774)/ (fzdepth**0.161))

def normalized_distance_term(x, intensity, fzdepth):
    return np.log(x/calculate_flamelength(intensity, fzdepth))

def slope_term(slope):
    return np.sin(2*np.radians(slope))**2

def wind_term(windspeed, intensity, fzdepth):
    return np.sin(np.arctan(1.4 * ((windspeed)**2 / (9.81 * calculate_flamelength(intensity, fzdepth)))**(1/2)))

def aspect_ratio_term(fzdepth, fzwidth):
    return fzdepth/fzwidth

def depth_by_fl_term(fzdepth, intensity):
    return fzdepth/calculate_flamelength(intensity, fzdepth)

def slope_wind_interact(slope, windspeed, intensity, fzdepth):
    return (slope_term(slope)* wind_term(windspeed, intensity, fzdepth))

def distance_slope_interact(x, intensity, fzdepth, slope):
    return (normalized_distance_term(x, intensity, fzdepth)*slope_term(slope))

def distance_wind_interact(x, intensity, fzdepth, windspeed):
    return (normalized_distance_term(x, intensity, fzdepth)*wind_term(windspeed, intensity, fzdepth))

def distance_aspect_ratio_interact(x, intensity, fzdepth, fzwidth):
    return (normalized_distance_term(x, intensity, fzdepth)*aspect_ratio_term(fzdepth, fzwidth))

def front_orientation_angle(windspeed, intensity, fzdepth, slope, angle):
    return np.cos(np.radians(angle)/ 2)**(wind_term(windspeed, intensity, fzdepth)+slope_term(slope))

def calculate_ln_temp(x, intensity, slope, angle, windspeed, fzdepth, fzwidth):
    '''
    calculates ln(temp) using our lab's updated statistical model
    ***alter coefficients here to update statistical model
    Inputs:
    x : x_points
    the rest are slider values
    
    Outputs: 
    one y output for given parameters
    '''        
    ln_temp = (-3.97429 +
               -.67645 * normalized_distance_term(x, intensity, fzdepth)
               +
               1.610607 * slope_term(slope)
               +
               1.488254 * wind_term(windspeed, intensity, fzdepth)
               +
               -.52375 * aspect_ratio_term(fzdepth, fzwidth)
               +    
               -.08213 * depth_by_fl_term(fzdepth, intensity)
               +
               -.66479 * slope_wind_interact(slope, windspeed, intensity, fzdepth)
               +
               -.52542 * distance_slope_interact(x, intensity, fzdepth, slope)
               +
               -.35344 * distance_wind_interact(x, intensity, fzdepth, windspeed)
               +
               0.134261 * distance_aspect_ratio_interact(x, intensity, fzdepth, fzwidth)
               +
               8.421952 * front_orientation_angle(windspeed, intensity, fzdepth, slope, angle)
               
    )
    return ln_temp

# def average_level_variable(level_variable : Static_Feature ):
#     '''
#     for averaging the statistical model input across its range when that variable is selected as the level variable. 
#     without this, the statistical model will use the greyed out slider variable when that input is the level variable
#     which is not preferable
#     '''
#     max = level_variable.get_max()
#     min = level_variable.get_min()
#     return (max+min)/2

# def alter_slider_for_stat_plot(slider_values: dict, level_variable ):
# 	'''
#     unnecessary now that we are not disabling the sliders. This was originally meant to 
# 	altering slider_values in this function and so that slider_values elsewhere is not mutated
# 	'''	
# 	slider_values_copy = slider_values.copy()
# 	for slider_value in slider_values_copy.keys():
# 		if slider_value == level_variable.short_name:
# 			slider_values_copy[slider_value] = average_level_variable(level_variable )
   
# 	return slider_values_copy

def filter_high_temps(pred_set: tuple):
	'''
 	for filtering all temps over 1000 C in plot outputs
	### Change filter temp in this function
	'''
	#making the tuple a list so that values can be assigned and then back to tuple
	pred_set_list = list(pred_set)
	for i in range(len(pred_set_list)):
		if pred_set_list[i] >= 1000:
			pred_set_list[i] = 1000
	return tuple(pred_set_list)

def fill_stat_y_set(x_points, slider_values):
    '''
    fills stat_y_set, using the altered_slider_values dict and filters out temps for the flaming zone
    '''
    intensity = slider_values.get('Intensity.kwm') 
    slope = slider_values.get('slope.deg') 
    angle = slider_values.get('Angle.deg')
    windspeed = slider_values.get('WindSpd.mps')
    fzdepth = slider_values.get('FlameDepth.m')
    fzwidth = slider_values.get('width.meters')
    
    stat_y_set = []
    for x in x_points:
        stat_y_set.append(np.exp(calculate_ln_temp(x, intensity, slope, angle, windspeed, fzdepth, fzwidth)))
        #filtering out values over 1000 C
        filtered_stat_y_set = filter_high_temps(stat_y_set)
        
    return filtered_stat_y_set

def graph_results(slider_values: dict, level_variable: Static_Feature, lnx, lny, x_range_min, x_range_max, x_value_sample_count, view_statistical_model, plot_w_level_variables, burn_lookup_mode, burn_selection, avg_ml_error, avg_stat_error):
    fig = go.Figure()
    if burn_lookup_mode:
        x_value_list, true_values, ml_pred, stat_pred_list = get_selected_burn_data(select_cfg(), burn_selection)
        fig.add_trace(go.Scatter(x = x_value_list, y= true_values, name = "True Burn Values", line=dict(dash='dashdot')))
        fig.add_trace(go.Scatter(x = x_value_list, y = ml_pred, name = f"ML Predictions, Avg Error: {avg_ml_error:.2g}", line=dict(color = 'black', dash='dash')))
        fig.add_trace(go.Scatter(x = x_value_list, y = stat_pred_list, name = f"Statistical Model Predictions, Avg Error: {avg_stat_error:.2g}", line=dict(color="black")))
    else:
        x_points = get_x_axis_points(x_range_min, x_range_max, x_value_sample_count)
        level_var_values = generate_level_variable_input_valuesII(level_variable, slider_values)
        y_set = get_prediction_output(level_variable, level_var_values, slider_values, x_points)

        stat_y_set = fill_stat_y_set(x_points, slider_values) 
        #create line
        fig.add_trace(go.Scatter(x=x_points, y=y_set[0].tolist(), name=f"Matches Statistical Model: {level_var_values[0]:.2f}", line=dict(color = 'black', dash='dash')))
        if plot_w_level_variables:
            fig.add_trace(go.Scatter(x=x_points, y=y_set[1].tolist(), name=f"{level_var_values[1]:.2f}", line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=x_points, y=y_set[2].tolist(), name=f"{level_var_values[2]:.2f}", line=dict(dash='dashdot')))
            fig.add_trace(go.Scatter(x=x_points, y=y_set[3].tolist(), name=f"{level_var_values[3]:.2f}"))
            fig.add_trace(go.Scatter(x=x_points, y=y_set[4].tolist(), name=f"{level_var_values[4]:.2f}", line=dict(dash='dash')))
   
        if view_statistical_model:
            fig.add_trace(go.Scatter(x=x_points, y=stat_y_set, name=f"Statistical Model: {level_var_values[0]:.2f}", line=dict(color="black")))

    #graph labels
    if burn_lookup_mode:
        fig.update_layout(title="Predictions vs True Values",
            xaxis_title="Distance (m)",
            yaxis_title="Temperature (C)",
            legend_title="Legend")
    else:
        fig.update_layout(title="Predictions",
            xaxis_title="Distance (m)",
            yaxis_title="Temperature (C)",
            legend_title=f"Level Variable:{level_variable.short_name}")

    #create log axes
    if lny == True:	
        fig.update_yaxes(type="log")
    if lnx == True:
        fig. update_xaxes(type="log")

    #creates graph on web page
    st.plotly_chart(fig, use_container_width=True)
    if burn_lookup_mode:
        data = get_csv_data(select_cfg())
        data_1 = data[data["burn"]==burn_selection]
        st.markdown(f'Intensity(Kwm)    :   :red[{data_1["Intensity.kwm"].iloc[0]:.3g}]  Slope(deg)     :   :red[{data_1["slope.deg"].iloc[0]:.2g}] Angle(deg)      :   :red[{data_1["Angle.deg"].iloc[0]:.2g}]  Windspeed(m/s)     :   :red[{data_1["WindSpd.mps"].iloc[0]:.2g}]  Flame Depth(m)   :  :red[{data_1["FlameDepth.m"].iloc[0]:.2g}]   Flame Width(m)  : :red[{data_1["width.meters"].iloc[0]:.2g}]')
    #nested list of y_values for .csv download
    return_list = []
    if burn_lookup_mode:
        return_list.append(x_value_list)
        return_list.append(true_values)
        return_list.append(ml_pred)
        return_list.append(stat_pred_list)
    else:
        return_list.append(x_points)
        i = 0
        while i < len(y_set):
            return_list.append(y_set[i])
            i += 1
    return return_list

def get_level_variable_index(selected_display_name):
    i = 0
    STATIC_FEATURE_LOOKUP_copy = STATIC_FEATURE_LOOKUP.copy()
    for static_feature in STATIC_FEATURE_LOOKUP_copy:
        if static_feature.display_name == selected_display_name:
            return i;
        i += 1

def get_all_display_names():
	display_names = [feature.display_name for feature in STATIC_FEATURE_LOOKUP]
	return display_names;

def make_st_slider(short_name, burn_lookup_mode : bool):
    static_feature = get_static_feature_from_short_name(short_name)

    # if the session_state doesn't have the static feature yet, then it gets the default
    # if the static feature is actually the level variable, disable the slider
    st.slider(
        static_feature.display_name,
        static_feature.gui_min,
        static_feature.max,
        st.session_state.get(short_name, static_feature.default),
        static_feature.step,
        key=short_name,
        disabled = burn_lookup_mode
        )

def get_static_feature_from_display_name(display_name) -> Static_Feature:
	return list(filter(lambda x: x.display_name == display_name, STATIC_FEATURE_LOOKUP))[0]

def get_static_feature_from_short_name(short_name) -> Static_Feature:
	return list(filter(lambda x: x.short_name == short_name, STATIC_FEATURE_LOOKUP))[0]

def create_gui():
	
    st.title("Carousel ML Gui")

    col0, col1 = st.columns(2, gap="medium")
    col2, col3, col4, col5  = st.columns(4, gap="large")
    row1 = col0, col1
    row2 = col2, col3

    display_names = get_all_display_names()
    
    display_names.remove("Flame Length (m)")
    display_names.remove("Flame Velocity (m/s)")

    with col0:
        burns_list = get_burn_names(get_csv_data(select_cfg()))
        burn_lookup_mode = st.toggle("Burn Lookup Mode", value = st.session_state.get("burn_lookup_mode", False), key = "burn_lookup_mode")
        current_selection_index = get_level_variable_index(st.session_state.get("level_var", "Angle (degrees)"))
        plot_w_level_variables = st.toggle("Plot Showing Level Variables", value = st.session_state.get("plot_w_level_variables", True), key = "plot_w_level_variables", disabled = burn_lookup_mode)
        view_statistical_model = st.toggle("View Statistical Model", value = st.session_state.get("view_statistical_model", True), key = "view_statistical_model", disabled= burn_lookup_mode)
        level_var_display_name = st.selectbox("Level Variable", display_names, key="level_var", disabled = burn_lookup_mode)
        level_variable = get_static_feature_from_display_name(level_var_display_name)
        make_st_slider("slope.deg", burn_lookup_mode)
        make_st_slider("Angle.deg", burn_lookup_mode)
        
    with col1:
        if burn_lookup_mode:
            data = get_csv_data(select_cfg())
            burn_selection = st.selectbox("Select Burn from Test Data", burns_list, key="burn_selection", disabled = not burn_lookup_mode)
            data_1 = data[data["burn"]==burn_selection]
            st.markdown("Parameters for Selected Test Burn:")
            st.markdown(f'Intensity(Kwm): :red[{data_1["Intensity.kwm"].iloc[0]:.3g}]   Slope(deg) : :red[{data_1["slope.deg"].iloc[0]:.2g}]   Angle(deg) : :red[{data_1["Angle.deg"].iloc[0]:.2g}]   Windspeed(m/s): :red[{data_1["WindSpd.mps"].iloc[0]:.2g}]   Flame Depth(m): :red[{data_1["FlameDepth.m"].iloc[0]:.2g}]   Flame Width(m): :red[{data_1["width.meters"].iloc[0]:.2g}]')
            x_value_list, true_values, ml_pred, stat_pred_list = get_selected_burn_data(select_cfg(),burn_selection)
            avg_ml_error, avg_stat_error = calculate_error_for_test_burns(get_csv_data(select_cfg()),burn_selection, true_values, ml_pred, stat_pred_list)
            st.markdown(f"Average Error of ML Predictions: :red[{avg_ml_error:.4g}]")
            st.markdown(f"Average Error of Statistical Model Predictions: :red[{avg_stat_error:.4g}]")
        else:
            # make_st_slider("FL(m)", burn_lookup_mode)
            make_st_slider("width.meters", burn_lookup_mode)
            make_st_slider("FlameDepth.m", burn_lookup_mode)
            make_st_slider("Intensity.kwm", burn_lookup_mode)
            make_st_slider("WindSpd.mps", burn_lookup_mode)
            # make_st_slider("FlameVel.mps", burn_lookup_mode)
    with col2:
        log_x = st.toggle("Log X-Axis", value=st.session_state.get("log_x", False), key="log_x")
        log_y = st.toggle("Log Y-Axis", value=st.session_state.get("log_y", False), key="log_y")
        x_value_sample_count = st.slider("Number of X-value Samples", 50, 500, st.session_state.get("x_value_sample_count", 200), 10, key="x_value_sample_count", disabled= burn_lookup_mode)
    with col3:
        xr_min = st.number_input("Minimum of X-axis Sample Range", 0.0001, 1.25, float(st.session_state.get("xr_min", 0.0001)), key="xr_min", disabled= burn_lookup_mode)
        xr_max = st.number_input("Maximum of X-axis Sample Range", 0.0001, 1.25, float(st.session_state.get("xr_max", 1.25)), key="xr_max", disabled= burn_lookup_mode)

    slider_values = {}
	# code just for debugging headlessly
    if not burn_lookup_mode:
        if len(st.session_state) == 0:
            for static_feature in STATIC_FEATURE_LOOKUP:
                slider_values[static_feature.short_name] = static_feature.default
        else:
            for static_feature in STATIC_FEATURE_LOOKUP:
                # slider_values[static_feature.short_name] = st.session_state[static_feature.short_name]
                try:
                    slider_values[static_feature.short_name] = st.session_state[static_feature.short_name]
                except KeyError:
                    slider_values[static_feature.short_name] = 0
        
    #in case burn_selection doesn't exist..
    if not burn_lookup_mode:
        burn_selection = None
        avg_ml_error = None
        avg_stat_error = None
        
    res = graph_results(slider_values, level_variable, log_x, log_y, xr_min, xr_max, x_value_sample_count, view_statistical_model, plot_w_level_variables, burn_lookup_mode, burn_selection, avg_ml_error, avg_stat_error)

    dat = np.array(res)
    # ASK: why are we transposing?
    new_dat = np.transpose(dat)
    if burn_lookup_mode:
        col_names = ["Distance (m)", "True Temp (C)", "ML Predictions (C)", "Statistical Model Predictions (C)"]
    else:
        col_names = ["Distance (m)", "Temp_1 [match]", "Temp_2", "Temp_3", "Temp_4", "Temp_5"]


    udf = pds.DataFrame(new_dat, columns=col_names)

    @st.cache_data
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    new_file = convert_df(udf)

    with col4:
        filename = st.text_input("Enter name of output file: ", "enter filename + .csv", key="filename")
        button = st.button("Download Data to CSV")
        if button:
            st.download_button("Download", data=new_file, file_name=str(filename), mime='text/csv')

create_gui()
