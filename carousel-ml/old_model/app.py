import streamlit as st
import tensorflow as tf
import os.path
import pandas as pds
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import math

#model setup
model_filename = 'caro_model_2'
model_path = (str(os.getcwd()) + '/' + model_filename)
model = tf.keras.models.load_model(model_path)


tf.compat.v1.enable_eager_execution()

xaxis_dict = {
	"Width (m)" : [0.5, 20.0, 9],
	"Slope (degrees)" : [0.0, 45.0, 9],
	"Angle (degrees)" : [0.0, 90.0, 9],
	"Intensity (kwm)" : [40.0, 1000.0, 9],
	"Wind Speed (m/s)" : [0.0, 6.0, 9],
	"Flame Depth (m)" : [0.1, 0.7, 9],
	"Distance (m)" : [0.0, 1.25, 9]
}

x_index_dict = {
	"Width (m)" : 0,
	"Slope (degrees)" : 1,
	"Angle (degrees)" : 2,
	"Intensity (kwm)" : 3,
	"Wind Speed (m/s)" : 4,
	"Flame Depth (m)" : 5,
	"Distance (m)" : 6
}

def basic_preds(input_list):
	feature_batch = tf.constant([input_list])
	prediction_as_tensor = model(feature_batch)
	prediction_as_list = prediction_as_tensor.numpy().tolist()[0]
	return prediction_as_list

#sets values for x_axis points
#inputs: range dictionary, variable selection name, number of points desired
def set_axis(ax_dict, name, num_steps):
	axis_points = []
	ax_min = ax_dict.get(name)[0]
	ax_max = ax_dict.get(name)[1]
	val = ax_min
	step = (ax_max-ax_min)/num_steps
	while len(axis_points) <= num_steps:
		axis_points.append(val)
		val += step
	return axis_points

#handles case of user selected domain
def new_set_axis(mini, maxi, num_steps):
	axis_points = []
	val = mini
	step = (maxi-mini)/num_steps
	while len(axis_points) < num_steps:
		axis_points.append(val) # round here
		val += step
	return axis_points


#returns nested list with different x-axis variable values to be used to make predictions
#axis_list is list of different value points for selected variable
#val_list is list of values to be fed to model to get predictions
def get_xaxis_points(axis_list, vals_list, index):
	pred_list = []
	for val in axis_list:
		new_val=vals_list.copy()
		new_val[index] = val
		pred_list.append(new_val)
	return pred_list

#takes nested lists from get_xaxis_points and feeds to ml model, returns list of predictions
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
		pred_list.append(float(tens_as_list[0]))

	return pred_list

#conveinent for making multiple data lines for level variable
#gets predictions from create_pred_set and returns list of y_values based on user selection to create single data line
#probably not strictly speaking necessary but here we are
def create_data_line(x_points, name_yaxis, y_dict):
	pred_nums = create_pred_set(x_points)
	return pred_nums

#returns nested list of y_list values for graphing lines with level variable
def add_level_vars(name_level, point_list, name_xaxis, x_dict_index, x_dict, xaxis_points):
	levelv_set = []
	y_set = []
	index_x = x_dict_index.get(str(name_xaxis))
	index_l = x_dict_index.get(str(name_level))
	levelv_points = set_axis(x_dict, name_level, 9) #int is number of lines drawn for level variable
	l_points_list = get_xaxis_points(levelv_points, point_list, index_l)
	
	#runs through all lines for level variable and changes x-axis variable to send to ml_model
	for axis in l_points_list:
		new_vals = axis.copy()
		new_list = get_xaxis_points(xaxis_points, new_vals, index_x)
		levelv_set.append(new_list)
	
	#appends y-variables for each level variable line
	for line in levelv_set:
		y_set.append(create_pred_set(line))

	return y_set

#does the cool stuff
def graph_results(widget_inputs, x_name, l_name, lnx, lny, xr_min, xr_max, max_vals):
	x_points = new_set_axis(xr_min, xr_max, max_vals)
	levelv_names = set_axis(xaxis_dict, l_name, 9)
	graph_x = get_xaxis_points(x_points, widget_inputs, x_index_dict.get(x_name))
	graph_y = create_pred_set(graph_x)
	y_set = add_level_vars(l_name, widget_inputs, x_name, x_index_dict, xaxis_dict, x_points)

	
			
	for nested in y_set:
		newy= nested.copy()
		i=0
		for point in newy:
			nested[i] = math.exp(point)
			i+=1

	fig = go.Figure()

	#create line
	fig.add_trace(go.Scatter(x=x_points, y=y_set[0], name=levelv_names[0], line=dict(dash='dash')))
	fig.add_trace(go.Scatter(x=x_points, y=y_set[1], name=levelv_names[1], line=dict(dash='dot')))
	fig.add_trace(go.Scatter(x=x_points, y=y_set[2], name=levelv_names[2], line=dict(dash='dashdot')))
	fig.add_trace(go.Scatter(x=x_points, y=y_set[3], name=levelv_names[3]))
	fig.add_trace(go.Scatter(x=x_points, y=y_set[4], name=levelv_names[4], line=dict(dash='dash')))
	fig.add_trace(go.Scatter(x=x_points, y=y_set[5], name=levelv_names[5], line=dict(dash='dot')))
	fig.add_trace(go.Scatter(x=x_points, y=y_set[6], name=levelv_names[6], line=dict(dash='dashdot')))
	fig.add_trace(go.Scatter(x=x_points, y=y_set[7], name=levelv_names[7]))
	fig.add_trace(go.Scatter(x=x_points, y=y_set[8], name=levelv_names[8], line=dict(dash='dash')))
	fig.add_trace(go.Scatter(x=x_points, y=y_set[9], name=levelv_names[9], line=dict(dash='dot')))

	#graph labels
	fig.update_layout(title="Predictions",
		xaxis_title=x_name,
		yaxis_title="Temperature",
		legend_title=l_name)

	if lny == True:	
		fig.update_yaxes(type="log")
	if lnx == True:
		fig. update_xaxes(type="log")

	#creates graph on web page
	st.plotly_chart(fig, use_container_width=True)

	return_list = []
	return_list.append(x_points)
	i = 0
	while i<len(y_set):
		return_list.append(y_set[i])
		i+=1
	return return_list


def create_gui():
	val_list = []
	st.set_page_config(page_title="Carousel ML", layout="wide")
	st.title("Carousel ML Gui")

	col0, col1 = st.columns(2, gap="medium")
	col2, col3, col4, col5  = st.columns(4, gap="large")
	row1 = col0, col1
	row2 = col2, col3

	xaxis_list = ["Width (m)", "Slope (degrees)", "Angle (degrees)", "Intensity (kwm)", "Wind Speed (m/s)", "Flame Depth (m)", "Distance (m)"]
	levelv_list = xaxis_list.copy()
	key_list = ["width", "slope", "angle", "intensity", "good_wind", "flame_depth", "distance"]
	
	if "xaxis" not in st.session_state:
		with col0:
			xaxis = st.selectbox("X-Axis Variable", xaxis_list, 6, key="xaxis")
			levelv = st.selectbox("Level Variable", levelv_list, 0, key="levelv")
			width = st.slider("Width (m)", 0.5, 20.0, 2.6, 0.1, key="width", disabled=True)
			slope = st.slider("Slope (degrees)", 0.0, 45.0, 5.0, 0.5, key="slope")
			angle = st.slider("Angle (degrees)", 0.0, 90.0, 20.0, 1.0, key="angle")
			
		with col1:
			intensity = st.slider("Intensity (kwm)", 40.0, 1000.0, 85.0, 5.0, key="intensity")
			good_wind = st.slider("Wind Speed (m/s)", 0.0, 6.0, 1.0, 0.1, key="good_wind")
			flame_depth = st.slider("Flame Depth (m)", 0.1, 0.7, 0.4, 0.01, key="flame_depth")
			distance = st.slider("Distance (m)", 0.0, 1.25, 0.5, 0.01, key="distance", disabled=True)
		
		with col2:
			log_x = st.toggle("Log X-Axis", key="log_x")
			log_y = st.toggle("Log Y-Axis", key="log_y")
			max_vals = st.slider("Number of X-value Samples", 50, 500, 200, 10, key="max_vals")

		with col3:
			xr_min = st.number_input("Minimum of X-axis Sample Range", 0.0, 1.25, key="xr_min")
			xr_max = st.number_input("Maximum of X-axis Sample Range", 0.0, 1.25, 1.25, key="xr_max")

	else:
		with col0:
			xaxis = st.selectbox("X-Axis Variable", xaxis_list, x_index_dict.get(st.session_state.xaxis), key="xaxis")
			levelv = st.selectbox("Level Variable", levelv_list, x_index_dict.get(st.session_state.levelv), key="levelv")
			if xaxis == "Width (m)" or levelv == "Width (m)":
				width = st.slider("Width (m)", 0.5, 20.0,float(st.session_state.width), 0.1, key="width", disabled=True)
			else:
				width = st.slider("Width (m)", 0.5, 20.0, float(st.session_state.width), 0.1, key="width")

			if xaxis == "Slope (degrees)" or levelv == "Slope (degrees)":
				slope = st.slider("Slope (degrees)", 0.0, 45.0, float(st.session_state.slope), 0.5, key="slope", disabled=True)
			else:
				slope = st.slider("Slope (degrees)", 0.0, 45.0, float(st.session_state.slope), 0.5, key="slope")

			if xaxis == "Angle (degrees)" or levelv == "Angle (degrees)":
				angle = st.slider("Angle (degrees)", 0.0, 90.0, float(st.session_state.angle), 1.0, key="angle", disabled=True)
			else:
				angle = st.slider("Angle (degrees)", 0.0, 90.0, float(st.session_state.angle), 1.0, key="angle")

		with col1:
			if xaxis == "Intensity (kwm)" or levelv == "Intensity (kwm)":
				intensity = st.slider("Intensity (kwm)", 40.0, 1000.0, float(st.session_state.intensity), 5.0, key="intensity", disabled=True)
			else:
				intensity = st.slider("Intensity (kwm)", 40.0, 1000.0, float(st.session_state.intensity), 5.0, key="intensity")

			if xaxis == "Wind Speed (m/s)" or levelv == "Wind Speed (m/s)":
				good_wind = st.slider("Wind Speed (m/s)", 0.0, 6.0, float(st.session_state.good_wind), 0.1, key="good_wind", disabled=True)
			else:
				good_wind = st.slider("Wind Speed (m/s)", 0.0, 6.0, float(st.session_state.good_wind), 0.1, key="good_wind")

			if xaxis == "Flame Depth (m)" or levelv == "Flame Depth(m)":
				flame_depth = st.slider("Flame Depth (m)", 0.1, 0.7, float(st.session_state.flame_depth), 0.01, key="flame_depth", disabled=True)
			else:
				flame_depth = st.slider("Flame Depth (m)", 0.1, 0.7, float(st.session_state.flame_depth), 0.01, key="flame_depth")

			if xaxis == "Distance (m)" or levelv == "Distance (m)":
				distance = st.slider("Distance (m)", 0.0, 1.25, float(st.session_state.distance), 0.01, key="distance", disabled=True)
			else:
				distance = st.slider("Distance (m)", 0.0, 1.25, float(st.session_state.distance), 0.01, key="distance")
	
		with col2:
			log_x = st.toggle("Log X-Axis", value=st.session_state.log_x, key="log_x")
			log_y = st.toggle("Log Y-Axis", value=st.session_state.log_y, key="log_y")
			max_vals = st.slider("Number of X-value Samples", 50, 500, st.session_state.get("max_vals"), 10, key="max_vals")

		with col3:
			xr_min = st.number_input("Minimum of X-axis Sample Range", float(xaxis_dict.get(xaxis)[0]), float(xaxis_dict.get(xaxis)[1]), key="xr_min")
			xr_max = st.number_input("Maximum of X-axis Sample Range", float(xaxis_dict.get(xaxis)[0]), float(xaxis_dict.get(xaxis)[1]), float(xaxis_dict.get(xaxis)[1]), key="xr_max")
			
	

	for key in key_list:
		val_list.append(st.session_state[key])

	res=graph_results(val_list, xaxis, levelv, log_x, log_y, xr_min, xr_max, max_vals)

	i=0

	dat = np.array(res)
	new_dat = np.transpose(dat)
	#print(new_dat)
	col_names = [xaxis, "Temp_1", "Temp_2", "Temp_3", "Temp_4", "Temp_5", "Temp_6", "Temp_7", "Temp_8", "Temp_9", "Temp_10"]

	udf = pds.DataFrame(new_dat, columns=col_names)
	
	@st.cache_data
	def convert_df(df):
		return df.to_csv().encode('utf-8')

	new_file = convert_df(udf)

	with col4:
		filename = st.text_input("Enter name of output file: ", key="filename")
		dnld = st.download_button("Download .csv", data=new_file, file_name=str(filename), mime='text/csv')

create_gui()