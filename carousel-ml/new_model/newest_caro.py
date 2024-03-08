import streamlit as st
import tensorflow as tf
import os.path
import pandas as pds
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import math

#model setup
model_filename = 'par_model_final/par_model' #change model here if needed
model_path = (str(os.getcwd()) + '/' + model_filename)
model = tf.keras.models.load_model(model_path)


tf.compat.v1.enable_eager_execution()

xaxis_dict = {
	"Width (m)" : [0.5, 20.0, 9],
	"Slope (degrees)" : [0.00001, 45.0, 9],
	"Angle (degrees)" : [0.00001, 90.0, 9],
	"Intensity (kwm)" : [40.0, 1000.0, 9],
	"Wind Speed (m/s)" : [0.00001, 6.0, 9],
	"Flame Depth (m)" : [0.1, 0.7, 9],
	"Distance (m)" : [0.00001, 1.25, 9]
}

x_index_dict = {
	"Width (m)" : 0,
	"Slope (degrees)" : 1,
	"Angle (degrees)" : 2,
	"Intensity (kwm)" : 3,
	"Wind Speed (m/s)" : 4,
	"Flame Depth (m)" : 5
}

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

#takes list of inputs and feeds to ml model, returns list of predictions
#I'm taking the log of the list of x-coordinates and feeding them into the linear equation produced by the model
#The e^(resultant y-valus) is what is displayed on the graph
def create_pred_set(single_list, xaxis_points):
	tensor_list = []
	pred_list = []
	in_tensor = tf.constant([single_list]) #creates list of tensor objects

	prediction_as_tensor = model.predict(in_tensor)

	tens_as_list = (prediction_as_tensor.tolist()[0]) #converts prediction tensors to lists
	ln_x = np.log(xaxis_points).tolist()

	for point in ln_x:
		pred_point = (float(tens_as_list[1])*point)+float(tens_as_list[0])

		pred_list.append(pred_point)

	exp_pred = np.exp(pred_list).tolist()
	return exp_pred


#conveinent for making multiple data lines for level variable
#gets predictions from create_pred_set and returns list of y_values based on user selection to create single data line
#probably not strictly speaking necessary but here we are
def create_data_line(x_points, name_yaxis, y_dict):
	pred_nums = create_pred_set(x_points)
	return pred_nums

#returns nested list of y_list values for graphing lines with level variable
def add_level_vars(name_level, point_list, x_dict_index, x_dict, xaxis_points):
	levelv_set = []
	y_set = []
	index_l = x_dict_index.get(str(name_level))
	levelv_points = set_axis(x_dict, name_level, 9) #int is number of lines drawn for level variable
	l_points_list = get_xaxis_points(levelv_points, point_list, index_l) #nested list containing user-inputs with stepped levelv values
	
	for line in l_points_list:
		y_set.append(create_pred_set(line, xaxis_points))
	return y_set

#creates the graph
def graph_results(widget_inputs, x_name, l_name, lnx, lny, xr_min, xr_max, max_vals):
	x_points = new_set_axis(xr_min, xr_max, max_vals)
	levelv_names = set_axis(xaxis_dict, l_name, 9)
	y_set = add_level_vars(l_name, widget_inputs, x_index_dict, xaxis_dict, x_points)

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
	
	#create log axes
	if lny == True:	
		fig.update_yaxes(type="log")
	if lnx == True:
		fig. update_xaxes(type="log")

	#creates graph on web page
	st.plotly_chart(fig, use_container_width=True)

	#nested list of y_values for .csv download
	return_list = []
	return_list.append(x_points)
	i = 0
	while i<len(y_set):
		return_list.append(y_set[i])
		i+=1
	return return_list


def create_gui():
	val_list = []
	st.set_page_config(page_title="Carousel ML", page_icon=":fire:", layout="wide")
	st.title("Carousel ML Gui")

	col0, col1 = st.columns(2, gap="medium")
	col2, col3, col4, col5  = st.columns(4, gap="large")
	row1 = col0, col1
	row2 = col2, col3

	xaxis_list = ["Width (m)", "Slope (degrees)", "Angle (degrees)", "Intensity (kwm)", "Wind Speed (m/s)", "Flame Depth (m)"]
	levelv_list = xaxis_list.copy()
	key_list = ["width", "slope", "angle", "intensity", "good_wind", "flame_depth"]
	
	if "levelv" not in st.session_state:
		with col0:
			levelv = st.selectbox("Level Variable", levelv_list, 0, key="levelv")
			width = st.slider("Width (m)", 0.5, 20.0, 2.6, 0.1, key="width", disabled=True)
			slope = st.slider("Slope (degrees)", 0.0, 45.0, 5.0, 0.5, key="slope")
			angle = st.slider("Angle (degrees)", 0.0, 90.0, 20.0, 1.0, key="angle")
			
		with col1:
			intensity = st.slider("Intensity (kwm)", 40.0, 1000.0, 85.0, 5.0, key="intensity")
			good_wind = st.slider("Wind Speed (m/s)", 0.0, 6.0, 1.0, 0.1, key="good_wind")
			flame_depth = st.slider("Flame Depth (m)", 0.1, 0.7, 0.4, 0.01, key="flame_depth")
		
		with col2:
			log_x = st.toggle("Log X-Axis", key="log_x")
			log_y = st.toggle("Log Y-Axis", key="log_y")
			max_vals = st.slider("Number of X-value Samples", 50, 500, 200, 10, key="max_vals")

		with col3:
			xr_min = st.number_input("Minimum of X-axis Sample Range", 0.0001, 1.25, key="xr_min")
			xr_max = st.number_input("Maximum of X-axis Sample Range", 0.0001, 1.25, 1.25, key="xr_max")

	else:
		with col0:
			levelv = st.selectbox("Level Variable", levelv_list, x_index_dict.get(st.session_state.levelv), key="levelv")
			if levelv == "Width (m)":
				width = st.slider("Width (m)", 0.5, 20.0,float(st.session_state.width), 0.1, key="width", disabled=True)
			else:
				width = st.slider("Width (m)", 0.5, 20.0, float(st.session_state.width), 0.1, key="width")

			if levelv == "Slope (degrees)":
				slope = st.slider("Slope (degrees)", 0.0, 45.0, float(st.session_state.slope), 0.5, key="slope", disabled=True)
			else:
				slope = st.slider("Slope (degrees)", 0.0, 45.0, float(st.session_state.slope), 0.5, key="slope")

			if levelv == "Angle (degrees)":
				angle = st.slider("Angle (degrees)", 0.0, 90.0, float(st.session_state.angle), 1.0, key="angle", disabled=True)
			else:
				angle = st.slider("Angle (degrees)", 0.0, 90.0, float(st.session_state.angle), 1.0, key="angle")

		with col1:
			if levelv == "Intensity (kwm)":
				intensity = st.slider("Intensity (kwm)", 40.0, 1000.0, float(st.session_state.intensity), 5.0, key="intensity", disabled=True)
			else:
				intensity = st.slider("Intensity (kwm)", 40.0, 1000.0, float(st.session_state.intensity), 5.0, key="intensity")

			if levelv == "Wind Speed (m/s)":
				good_wind = st.slider("Wind Speed (m/s)", 0.0, 6.0, float(st.session_state.good_wind), 0.1, key="good_wind", disabled=True)
			else:
				good_wind = st.slider("Wind Speed (m/s)", 0.0, 6.0, float(st.session_state.good_wind), 0.1, key="good_wind")

			if levelv == "Flame Depth(m)":
				flame_depth = st.slider("Flame Depth (m)", 0.1, 0.7, float(st.session_state.flame_depth), 0.01, key="flame_depth", disabled=True)
			else:
				flame_depth = st.slider("Flame Depth (m)", 0.1, 0.7, float(st.session_state.flame_depth), 0.01, key="flame_depth")
	
		with col2:
			log_x = st.toggle("Log X-Axis", value=st.session_state.log_x, key="log_x")
			log_y = st.toggle("Log Y-Axis", value=st.session_state.log_y, key="log_y")
			max_vals = st.slider("Number of X-value Samples", 50, 500, st.session_state.get("max_vals"), 10, key="max_vals")

		with col3:
			xr_min = st.number_input("Minimum of X-axis Sample Range", 0.0001, 1.25, float(st.session_state.xr_min), key="xr_min")
			xr_max = st.number_input("Maximum of X-axis Sample Range", 0.0001, 1.25, float(st.session_state.xr_max), key="xr_max")
			
	

	for key in key_list:
		val_list.append(st.session_state[key])

	res=graph_results(val_list, "Distance (m)", levelv, log_x, log_y, xr_min, xr_max, max_vals)

	dat = np.array(res)
	new_dat = np.transpose(dat)
	#print(new_dat)
	col_names = ["Distance (m)", "Temp_1", "Temp_2", "Temp_3", "Temp_4", "Temp_5", "Temp_6", "Temp_7", "Temp_8", "Temp_9", "Temp_10"]

	udf = pds.DataFrame(new_dat, columns=col_names)
	
	@st.cache_data
	def convert_df(df):
		return df.to_csv().encode('utf-8')

	new_file = convert_df(udf)

	with col4:
		filename = st.text_input("Enter name of output file: ", key="filename")
		dnld = st.download_button("Download .csv", data=new_file, file_name=str(filename), mime='text/csv')

create_gui()
