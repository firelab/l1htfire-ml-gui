#streamlit version of spread gui
#Hannah Pinkerton, hannah.pinkerton@usda.gov
#Missoula Fire Lab, February 2024


import streamlit as st
import tensorflow as tf
import pandas as pds
import plotly.express as px
import plotly.graph_objects as go
import numpy
import os.path
import time

#lookup dictionaries
xaxis_dict = {
	'Bed Slope Angle (degrees)' : [0.0, 30.0, 0.3],
	'Bed Width (m)' : [1.0, 100.0, 0.5],
	'Fuel Clump Size (m)' : [0.5 , 2.0, 0.01],
	'Fuel Depth (m)' : [0.05, 1.0, 0.01],
	'Fuel Gap Size (m)' : [0.1, 0.5, 0.004],
	'Fuel Loading (kg/m^3)' : [0.05, 3.0, 0.05],
	'Ignition Depth (m)' : [0.1, 4.0, 0.02],
	'Particle Diameter (mm)' : [0.0005, 0.005, 0.00002],
	'Particle Moisture (%)' : [2.0, 35.0, 0.25],
	'Wind Amplitude (fraction)' : [0.2, 1.0, 0.01],
	'Mean Wind Speed (m/s)' : [0.0, 10.0, 0.1],
	'Wind Period (s)' : [1.0, 5.0, 0.04]
}

yaxis_dict = {
	'Spread Rate (m/min)' : [-1, 1, 4],
	'Flame Zone Depth (m)' : [-1, 1, 4],
	'Flame Length (m)' : [-1, 1, 4]
}

levelv_dict = {
	'Bed Slope Angle (degrees)' : [0, 30, 9],
	'Bed Width (m)' : [1, 100, 9],
	'Fuel Clump Size (m)' : [0.5 , 2, 9],
	'Fuel Depth (m)' : [0.05, 1, 9],
	'Fuel Gap Size (m)' : [0.1, 0.5, 9],
	'Fuel Loading (kg/m^3)' : [0.05, 3, 9],
	'Ignition Depth (m)' : [0.1, 4, 9],
	'Particle Diameter (mm)' : [.0005, .005, 9],
	'Particle Moisture (%)' : [2, 35, 9],
	'Wind Amplitude (fraction)' : [0.2, 1, 9],
	'Mean Wind Speed (m/s)' : [0, 10, 9],
	'Wind Period (s)' : [1, 5, 9]
}

#value int corresponds to key's index in point_list
x_index_dict = {
	'Bed Slope Angle (degrees)' : 0,
	'Bed Width (m)' : 1,
	'Fuel Clump Size (m)' : 2,
	'Fuel Depth (m)' : 3,
	'Fuel Gap Size (m)' : 4,
	'Fuel Loading (kg/m^3)' : 5,
	'Ignition Depth (m)' : 6,
	'Particle Diameter (mm)' : 7,
	'Particle Moisture (%)' : 8,
	'Wind Amplitude (fraction)' : 9,
	'Mean Wind Speed (m/s)' : 10,
	'Wind Period (s)' : 11,
}

y_index_dict = {
	'Flame Length (m)' : 0,
	'Flame Zone Depth (m)' : 1,
	'Spread Rate (m/min)' : 2
}

fcont_index_dict = {
	'Continuous' : 0,
	'Discontinuous' : 1
}

wc_index_dict = {
	'Constant' : 0,
	'Sine' : 1
}

#model setup
model_filename = 'combined_autonorm_r2v4.h5'
model_path = (str(os.getcwd()) + '/' + model_filename)
model = tf.keras.models.load_model(model_path)

tf.compat.v1.enable_eager_execution()

#sets values for x_axis points
#inputs: range dictionary, variable selection name, number of points desired
def set_axis(ax_dict, name, num_steps):
	axis_points = []
	ax_min = ax_dict.get(name)[0]
	ax_max = ax_dict.get(name)[1]
	val = ax_min
	step = (ax_max-ax_min)/num_steps
	while len(axis_points) <= num_steps:
		axis_points.append(val) #round here
		val += step
	return axis_points

#handles case of user selected domain
def new_set_axis(mini, maxi, num_steps):
	axis_points = []
	val = mini
	step = (maxi-mini)/num_steps
	while len(axis_points) <= num_steps:
		axis_points.append(val) # round here
		val += step
	return axis_points

def basic_preds(input_list):
	feature_batch = tf.constant([input_list])
	prediction_as_tensor = model(feature_batch)
	prediction_as_list = prediction_as_tensor.numpy().tolist()[0]
	return prediction_as_list

#sets ranges for variables for graph widget
def set_range(ax_dict, name):
	ax_range = []
	ax_range.append(name)
	ax_range.append(ax_dict.get(name)[0])
	ax_range.append(ax_dict.get(name)[1])
	return ax_range

#returns nested list with different x-axis variable values to be used to make predictions
#axis_list is list of different value points for selected variable
#val_list is list of values to be fed to model to get predictions
def get_xaxis_points(axis_list, val_list, index):
	pred_list = []
	for val in axis_list:
		new_val=val_list.copy()
		new_val[index] = val
		pred_list.append(new_val)
	return pred_list

#takes nested lists from get_xaxis_points and feeds to ml model, returns nested list with predictions
def create_pred_set(nest_list):
	tensor_list = []
	pred_list = []
	for nested in nest_list:
		in_tensor = tf.constant([nested]) #creates list of tensor objects
		tensor_list.append(in_tensor)
	fin_tensor = tf.experimental.numpy.vstack(tensor_list)

	prediction_as_tensor = model.predict(fin_tensor)

	for tens in prediction_as_tensor:
		tens_as_list = (tens.tolist()) #converts prediction tensors to lists
		pred_list.append(tens_as_list)

	return pred_list

#takes lists of pred values and returns list of values for y-axis selection
def set_y_nums(name, pred_vals, sel_dict):
	index = sel_dict.get(name)
	y_list = []
	for nested in pred_vals:
		y_list.append(nested[index])#roudn ehre
	return y_list


#conveinent for making multiple data lines for level variable
#gets predictions from create_pred_set and returns list of y_values based on user selection to create single data line
#probably not strictly speaking necessary but here we are
def create_data_line(x_points, name_yaxis, y_dict):
	pred_nums = create_pred_set(x_points)
	y_list = set_y_nums(name_yaxis, pred_nums, y_dict)
	return y_list

#returns nested list of y_list values for graphing lines with level variable
def add_level_vars(name_level, point_list, name_xaxis, name_yaxis, x_dict_index, y_dict_index, l_dict, xaxis_points):
	levelv_set = []
	y_set = []
	index_x = x_dict_index.get(str(name_xaxis))
	index_l = x_dict_index.get(str(name_level))
	levelv_points = set_axis(l_dict, name_level, 9) #int is number of lines drawn for level variable
	l_points_list = get_xaxis_points(levelv_points, point_list, index_l)
	
	#runs through all lines for level variable and changes x-axis variable to send to ml_model
	for axis in l_points_list:
		new_vals = axis.copy()
		new_list = get_xaxis_points(xaxis_points, new_vals, index_x)
		levelv_set.append(new_list)
	
	#appends y-variables for each level variable line
	for line in levelv_set:
		y_set.append(create_data_line(line, name_yaxis, y_dict_index))

	return y_set	

def create_gui():	
	#image paths, change here for new favicon
	favicon_path = (str(os.getcwd() + '/static/favicon_new.ico'))
	fs_image_path = (str(os.getcwd()) + '/static/fs.png')
	google_image_path = (str(os.getcwd()) + '/static/google_research.png')

	#app setup
	st.set_page_config(page_title="L1HTFire-ML", page_icon=favicon_path, layout="wide")

	st.image(fs_image_path, width=100)
	st.image(google_image_path, width=400)

	st.title("L1HTFire-ML")


	intro = "The L1HTFire model (Linear 1-Dimensional Heat Transfer) is a physics-based wildland fire spread model created by the USFS Missoula Fire Sciences Laboratory. This is a Machine Learning (ML) surrogate of L1HTFire and displays three output variables: fire spread rate, flame zone depth, and flame length. **Google Research** created the ML version using 575 million training runs of the physical model. The ML surrogate model is hosted here for demonstration purposes only in order to illustrate how wildland fire behavior results from the complex relationships between environmental and fire variables. A full description of this model can be found in Wildland Fire Behaviour (Finney, M.A., McAllister, S.S., Forthofer, J.M. and Grumstrup, T.P., 2021. *Wildland Fire Behaviour: Dynamics, Principles and Processes.* CSIRO PUBLISHING.)"

	st.markdown(intro)

	range_help = "Select the minimum and maximum x-axis sample values."
	max_help = "Select the number of values to be sampled for the range specified above."

	#columns
	col0, col1, col2, col3 = st.columns(4, gap="medium")

	key_list = ["bsa", "bw", "fcs", "fd", "fgs", "fl", "igd", "pd", "pm", "wa", "mws", "wp", "wc", "fcont", "xaxis", "yaxis", "levelv", "max_vals"]
	value_list = []

	#lists for dropdown menu
	xaxis_list = ["Bed Slope Angle (degrees)", "Bed Width (m)", "Fuel Clump Size (m)", "Fuel Depth (m)", "Fuel Gap Size (m)", "Fuel Loading (kg/m^3)", "Ignition Depth (m)", "Particle Diameter (mm)", "Particle Moisture (%)", "Wind Amplitude (fraction)", "Mean Wind Speed (m/s)", "Wind Period (s)"]
	yaxis_list = ["Flame Length (m)", "Flame Zone Depth (m)", "Spread Rate (m/min)"]
	levelv_list = xaxis_list.copy()

	#initialize app with widgets
	if ("xaxis" not in st.session_state and len(st.query_params.to_dict()) == 0) or "xr_vals" not in st.query_params:

		with col0:

			xaxis = st.selectbox("X-axis Variable", xaxis_list, 10, key="xaxis")
			yaxis = st.selectbox("Y-axis Variable", yaxis_list, 2, key="yaxis")
			levelv = st.selectbox("Level Variable", levelv_list, 8, key="levelv")

			mws = st.slider("Mean Wind Speed (m/s)", 0.0, 10.0, 4.0, 0.1, disabled=True, key="mws")
			bsa = st.slider("Bed Slope Angle (degrees)", 0.0, 30.0, 0.0, 0.3, key="bsa")

		with col1:

			bw = st.slider("Bed Width (m)", 1.0, 100.0, 50.0, 0.5, key="bw")
			fd = st.slider("Fuel Depth (m)", 0.05, 1.0, 0.2, 0.01, key="fd")
			fl = st.slider("Fuel Loading (kg/m^3)", 0.05, 3.0, 0.2, 0.05, key="fl")
			igd = st.slider("Ignition Depth (m)", 0.1, 4.0, 2.0, 0.02, key="igd")
			pd = st.slider("Particle Diameter (mm)", 0.5, 5.0, 2.0, 0.02, key="pd")

		with col2:
			
			pm = st.slider("Particle Moisture (%)", 2.0, 35.0, 10.0, 0.25, disabled=True, key="pm")
			wc = st.radio("Wind Case", ["Constant", "Sine"], 0, key="wc")
			wa = st.slider("Wind Amplitude Relative to Mean (fraction)", 0.2, 1.0, 0.2, 0.01, disabled=True, key="wa")
			wp = st.slider("Wind Period (s)", 1.0, 5.0, 1.0, 0.04, disabled=True, key="wp")
			wc_code = 0

		with col3:

			fcont = st.radio("Fuel Continuity", ["Continuous", "Discontinuous"], 0, key="fcont")
			fcs = st.slider("Fuel Clump Size (m)", 0.5, 2.0, 0.5, 0.01, disabled=True, key="fcs")
			fgs = st.slider("Fuel Gap Size (m)", 0.1, 0.5, 0.1, 0.004, disabled=True, key="fgs")
			fc_code = 0

		x_range_vals = st.slider("Range of X-value Samples", 0.0, 10.0, [0.0, 10.0], 0.1, key="xr_vals", help=range_help)
		max_vals = st.slider("Number of X-value Samples", 50, 500, 200, 10, key="max_vals", help=max_help)
	
	#every other update to web page			
	else:
		#handles case where user has copied url; restartes page but uses url parameters for defaults
		if "xaxis" not in st.session_state:
			param_dict = st.query_params.to_dict()
		else:
			param_dict = st.session_state
		with col0:

			xaxis = st.selectbox("X-axis Variable", xaxis_list, x_index_dict.get(param_dict.get("xaxis")), key="xaxis")
			yaxis = st.selectbox("Y-axis Variable", yaxis_list, y_index_dict.get(param_dict.get("yaxis")), key="yaxis")
			levelv = st.selectbox("Level Variable", levelv_list, x_index_dict.get(param_dict.get("levelv")), key="levelv")

			#disables widgets based on axis and conditional selections
			if xaxis == "Mean Wind Speed (m/s)" or levelv == "Mean Wind Speed (m/s)":
				mws = st.slider("Mean Wind Speed (m/s)", 0.0, 10.0, float(param_dict.get("mws")), 0.1, disabled=True, key="mws")
			else:
				mws = st.slider("Mean Wind Speed (m/s)", 0.0, 10.0, float(param_dict.get("mws")), 0.1, key="mws")

			if xaxis == "Bed Slope Angle (degrees)" or levelv == "Bed Slope Angle (degrees)":
				bsa = st.slider("Bed Slope Angle (degrees)", 0.0, 30.0, float(param_dict.get("bsa")), 0.3, disabled=True, key="bsa")
			else:
				bsa = st.slider("Bed Slope Angle (degrees)", 0.0, 30.0, float(param_dict.get("bsa")), 0.3, key="bsa")

		with col1:

			if xaxis == "Bed Width (m)" or levelv == "Bed Width (m)":
				bw = st.slider("Bed Width (m)", 1.0, 100.0, float(param_dict.get("bw")), 0.5, disabled=True, key="bw")
			else:
				bw = st.slider("Bed Width (m)", 1.0, 100.0, float(param_dict.get("bw")), 0.5, key="bw")

			if xaxis == "Fuel Depth (m)" or levelv == "Fuel Depth (m)":
				fd = st.slider("Fuel Depth (m)", 0.05, 1.0, float(param_dict.get("fd")), 0.01, disabled=True, key="fd")
			else:
				fd = st.slider("Fuel Depth (m)", 0.05, 1.0, float(param_dict.get("fd")), 0.01, key="fd")

			if xaxis == "Fuel Loading (kg/m^3)" or levelv == "Fuel Loading (kg/m^3)":
				fl = st.slider("Fuel Loading (kg/m^3)", 0.05, 3.0, float(param_dict.get("fl")), 0.05, disabled=True, key="fl")
			else:
				fl = st.slider("Fuel Loading (kg/m^3)", 0.05, 3.0, float(param_dict.get("fl")), 0.05, key="fl")

			if xaxis == "Ignition Depth (m)" or levelv == "Ignition Depth (m)":
				igd = st.slider("Ignition Depth (m)", 0.1, 4.0, float(param_dict.get("igd")), 0.02, disabled=True, key="igd")
			else:
				igd = st.slider("Ignition Depth (m)", 0.1, 4.0, float(param_dict.get("igd")), 0.02, key="igd")
			
			if xaxis == "Particle Diameter (mm)" or levelv == "Particle Diameter (mm)":
				pd = st.slider("Particle Diameter (mm)", 0.5, 5.0, float(param_dict.get("pd")), 0.02, disabled=True, key="pd")
			else:
				pd = st.slider("Particle Diameter (mm)", 0.5, 5.0, float(param_dict.get("pd")), 0.02, key="pd")

		with col2:

			if xaxis == "Particle Moisture (%)" or levelv == "Particle Moisture (%)":
				pm = st.slider("Particle Moisture (%)", 2.0, 35.0, float(param_dict.get("pm")), 0.25, disabled=True, key="pm")
			else:
				pm = st.slider("Particle Moisture (%)", 2.0, 35.0, float(param_dict.get("pm")), 0.25, key="pm")

			if xaxis == "Wind Amplitude (fraction)" or xaxis == "Wind Period (s)" or levelv == "Wind Amplitude (fraction)" or levelv == "Wind Period (s)":
				wc = st.radio("Wind Case", ["Constant", "Sine"], 1, disabled=True, key="wc")
			else:
				wc = st.radio("Wind Case", ["Constant", "Sine"], int(wc_index_dict.get(param_dict.get("wc"))), key="wc")

			#logic to disable/enable sliders for wind
			if wc == "Constant" or ((xaxis == "Wind Amplitude (fraction)" or levelv == "Wind Amplitude (fraction)") and (xaxis == "Wind Period (s)" or levelv == "Wind Period (s)")):
				wa = st.slider("Wind Amplitude Relative to Mean (fraction)", 0.2, 1.0, float(param_dict.get("wa")), 0.01, disabled=True, key="wa")
				wp = st.slider("Wind Period (s)", 1.0, 5.0, float(param_dict.get("wp")), 0.04, disabled=True, key="wp")
			elif xaxis == "Wind Amplitude (fraction)" or levelv == "Wind Amplitude (fraction)":
				wa = st.slider("Wind Amplitude Relative to Mean (fraction)", 0.2, 1.0, float(param_dict.get("wa")), 0.01, disabled=True, key="wa")
				wp = st.slider("Wind Period (s)", 1.0, 5.0, float(param_dict.get("wp")), 0.04, key="wp")
	
			elif xaxis == "Wind Period (s)" or levelv == "Wind Period (s)":
				wa = st.slider("Wind Amplitude Relative to Mean (fraction)", 0.2, 1.0, float(param_dict.get("wa")), 0.01, key="wa")
				wp = st.slider("Wind Period (s)", 1.0, 5.0, float(param_dict.get("wp")), 0.04, disabled=True, key="wp")
			else:
				wa = st.slider("Wind Amplitude Relative to Mean (fraction)", 0.2, 1.0, float(param_dict.get("wa")), 0.01, key="wa")
				wp = st.slider("Wind Period (s)", 1.0, 5.0, float(param_dict.get("wp")), 0.04, key="wp")

		with col3:
			#does not allow sine wind with continuous fuel (model not trained on case)
			#logic to disable/enable sliders/radio for fuel continuity
			if wc == "Sine": 
				fcont = st.radio("Fuel Continuity", ["Continuous", "Discontinuous"], 1, disabled=True, key="fcont")
			elif xaxis == "Fuel Clump Size (m)" or xaxis == "Fuel Gap Size (m)" or levelv == "Fuel Clump Size (m)" or levelv == "Fuel Gap Size (m)":
				fcont = st.radio("Fuel Continuity", ["Continuous", "Discontinuous"], 1, disabled=True, key="fcont")
			else:
				fcont = st.radio("Fuel Continuity", ["Continuous", "Discontinuous"], int(fcont_index_dict.get(param_dict.get("fcont"))), key="fcont")

			if (wc == "Sine" and (xaxis == "Fuel Gap Size (m)" or levelv == "Fuel Gap Size (m)")):
				fcs = st.slider("Fuel Clump Size (m)", 0.5, 2.0, float(param_dict.get("fcs")), 0.01, key="fcs")
				fgs = st.slider("Fuel Gap Size (m)", 0.1, 0.5, float(param_dict.get("fgs")), 0.004, disabled=True, key="fgs")				
			elif (wc == "Sine" and (xaxis == "Fuel Clump Size (m)" or levelv == "Fuel Clump Size (m)")):
				fcs = st.slider("Fuel Clump Size (m)", 0.5, 2.0, float(param_dict.get("fcs")), 0.01, disabled=True, key="fcs")
				fgs = st.slider("Fuel Gap Size (m)", 0.1, 0.5, float(param_dict.get("fgs")), 0.004, key="fgs")				
			elif (wc == "Sine"):
				fcs = st.slider("Fuel Clump Size (m)", 0.5, 2.0, float(param_dict.get("fcs")), 0.01, key="fcs")
				fgs = st.slider("Fuel Gap Size (m)", 0.1, 0.5, float(param_dict.get("fgs")), 0.004, key="fgs")
			elif fcont == "Continuous" or ((xaxis == "Fuel Clump Size (m)" or levelv == "Fuel Clump Size (m)") and (xaxis == "Fuel Gap Size (m)" or levelv == "Fuel Gap Size (m)")):
				fcs = st.slider("Fuel Clump Size (m)", 0.5, 2.0, float(param_dict.get("fcs")), 0.01, disabled=True, key="fcs")
				fgs = st.slider("Fuel Gap Size (m)", 0.1, 0.5, float(param_dict.get("fgs")), 0.004, disabled=True, key="fgs")
			elif xaxis == "Fuel Clump Size (m)" or levelv == "Fuel Clump Size (m)":
				fcs = st.slider("Fuel Clump Size (m)", 0.5, 2.0, float(param_dict.get("fcs")), 0.01, disabled=True, key="fcs")
				fgs = st.slider("Fuel Gap Size (m)", 0.1, 0.5, float(param_dict.get("fgs")), 0.004, key="fgs")
			elif xaxis == "Fuel Gap Size (m)" or levelv == "Fuel Gap Size (m)":
				fcs = st.slider("Fuel Clump Size (m)", 0.5, 2.0, float(param_dict.get("fcs")), 0.01, key="fcs")
				fgs = st.slider("Fuel Gap Size (m)", 0.1, 0.5, float(param_dict.get("fgs")), 0.004, disabled=True, key="fgs")
			else:
				fcs = st.slider("Fuel Clump Size (m)", 0.5, 2.0, float(param_dict.get("fcs")), 0.01, key="fcs")
				fgs = st.slider("Fuel Gap Size (m)", 0.1, 0.5, float(param_dict.get("fgs")), 0.004, key="fgs")

		if 'xr_vals' not in st.session_state and xaxis != "Particle Diameter (mm)":
			min_max = st.query_params.get_all('xr_vals')
			x_range_vals = st.slider("Range of X-value Samples", float(xaxis_dict.get(xaxis)[0]), float(xaxis_dict.get(xaxis)[1]), [float(min_max[0]), float(min_max[1])], float(xaxis_dict.get(xaxis)[2]), key="xr_vals", help=range_help)
		elif 'xr_vals' not in st.session_state and xaxis == "Particle Diameter (mm)":
			min_max = st.query_params.get_all('xr_vals')
			x_range_vals = st.slider("Range of X-value Samples", float(xaxis_dict.get(xaxis)[0])*1000, float(xaxis_dict.get(xaxis)[1])*1000, [float(min_max[0]), float(min_max[1])], float(xaxis_dict.get(xaxis)[2])*1000, key="xr_vals", help=range_help)
		elif xaxis == "Particle Diameter (mm)":
			x_range_vals = st.slider("Range of X-value Samples", float(xaxis_dict.get(xaxis)[0])*1000, float(xaxis_dict.get(xaxis)[1])*1000, [float(xaxis_dict.get(xaxis)[0])*1000, float(xaxis_dict.get(xaxis)[1])*1000], float(xaxis_dict.get(xaxis)[2])*1000, key="xr_vals", help=range_help)
		#ha checks if xaxis has changed-- session_state updates query params at end of run
		elif str(st.query_params.xaxis) == str(param_dict.get("xaxis")):
			min_max = param_dict.get('xr_vals')
			x_range_vals = st.slider("Range of X-value Samples", float(xaxis_dict.get(xaxis)[0]), float(xaxis_dict.get(xaxis)[1]), [float(min_max[0]), float(min_max[1])], float(xaxis_dict.get(xaxis)[2]), key="xr_vals", help=range_help)
		else:
			x_range_vals = st.slider("Range of X-value Samples", float(xaxis_dict.get(xaxis)[0]), float(xaxis_dict.get(xaxis)[1]), [float(xaxis_dict.get(xaxis)[0]), float(xaxis_dict.get(xaxis)[1])], float(xaxis_dict.get(xaxis)[2]), key="xr_vals", help=range_help)
		
		max_vals = st.slider("Number of X-value Samples", 50, 500, int(param_dict.get("max_vals")), 10, key="max_vals", help=max_help)

	#handles wind case/fuel continuity cases	
	if wc == "Constant":
		wc_code = 0
		wa = 0.0
		wp = 0.0
	else:
		wc_code = 1

	if fcont == "Continuous":
		fc_code = 0
		fcs = 0.0
		fgs = 0.0
	else:
		fc_code = 1

	#values from user input sent to model
	value_list.append(bsa)
	value_list.append(bw)
	value_list.append(fcs)
	value_list.append(fd)
	value_list.append(fgs)
	value_list.append(fl)
	value_list.append(igd)
	value_list.append(pd/1000) #particle diameter needs to be in meters to go to model
	value_list.append(pm)
	value_list.append(wa)
	value_list.append(mws)
	value_list.append(wp)
	value_list.append(wc_code)

	#retrieves model data
	if xaxis == "Particle Diameter (mm)":
		xaxis_points = new_set_axis(x_range_vals[0]/1000, x_range_vals[1]/1000, max_vals-1) #number of points on graph
	else:
		xaxis_points = new_set_axis(x_range_vals[0], x_range_vals[1], max_vals-1) #number of points on graph
		
	model_x = get_xaxis_points(xaxis_points, value_list, x_index_dict.get(xaxis))
	y_set = (add_level_vars(str(levelv), value_list, str(xaxis), str(yaxis), x_index_dict, y_index_dict, levelv_dict, xaxis_points))	
	levelv_names = set_axis(levelv_dict, levelv, 9)

	#returns particle diameter to mm
	if str(levelv) == "Particle Diameter (mm)":
		i=0
		while i < len(levelv_names):
			mult_level = levelv_names.copy()
			levelv_names[i] = mult_level[i]*1000
			i+=1

	if str(xaxis) == "Particle Diameter (mm)":
		i=0
		while i < len(xaxis_points):
			mult_xaxis = xaxis_points.copy()
			xaxis_points[i] = mult_xaxis[i]*1000
			i+=1

	j = 0
	while j < len(levelv_names):
		conc_level = levelv_names.copy()
		levelv_names[j] = round(conc_level[j],2)
		j+=1

	#graph
	fig = go.Figure()

	#create lines
	fig.add_trace(go.Scatter(x=xaxis_points, y=y_set[0], name=levelv_names[0], line=dict(dash='dash')))
	fig.add_trace(go.Scatter(x=xaxis_points, y=y_set[1], name=levelv_names[1], line=dict(dash='dot')))
	fig.add_trace(go.Scatter(x=xaxis_points, y=y_set[2], name=levelv_names[2], line=dict(dash='dashdot')))
	fig.add_trace(go.Scatter(x=xaxis_points, y=y_set[3], name=levelv_names[3]))
	fig.add_trace(go.Scatter(x=xaxis_points, y=y_set[4], name=levelv_names[4], line=dict(dash='dash')))
	fig.add_trace(go.Scatter(x=xaxis_points, y=y_set[5], name=levelv_names[5], line=dict(dash='dot')))
	fig.add_trace(go.Scatter(x=xaxis_points, y=y_set[6], name=levelv_names[6], line=dict(dash='dashdot')))
	fig.add_trace(go.Scatter(x=xaxis_points, y=y_set[7], name=levelv_names[7]))
	fig.add_trace(go.Scatter(x=xaxis_points, y=y_set[8], name=levelv_names[8], line=dict(dash='dash')))
	fig.add_trace(go.Scatter(x=xaxis_points, y=y_set[9], name=levelv_names[9], line=dict(dash='dot')))

	fig.update_traces(hovertemplate = "(%{x: .4r}, %{y: .4r})")

	#graph labels
	fig.update_layout(title="Predictions",
		xaxis_title=xaxis,
		yaxis_title=yaxis,
		legend_title=levelv,
		height=800)

	#creates graph on web page
	st.plotly_chart(fig, use_container_width=True)

	#preparation for url parameters
	vals = value_list.copy()
	vals.append(fc_code)
	vals.append(xaxis)
	vals.append(yaxis)
	vals.append(levelv)

	#get all user inputs in dictionary
	param_dict = st.session_state

	#clear url parameters and set parameters to proper key/value pairs
	st.query_params.clear()
	for key in param_dict.keys():
		st.query_params[key] = param_dict.get(key)

	return value_list

create_gui()
