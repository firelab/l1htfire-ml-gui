#web_gui with integreted tensorflow/ml_model
#Pinkerton, Missoula Fire Lab, Jan 2024
#to be deployed to web server

from flask import Flask
from flask import request, render_template, flash, url_for, redirect
import tensorflow as tf
import numpy
import os.path

#model setup
model_filename = 'ml_model'
model_path = (str(os.getcwd()) + '/' + model_filename)
model = tf.keras.models.load_model(model_path)
print('model loaded')

feature_names = ['bed_slope_angle', 'bed_width', 'fuel_clump_size', 'fuel_depth', 'fuel_gap_size', 'fuel_loading', 'ignition_depth', 'particle_diameter', 'particle_moisture', 'wind_amplitude_rel_mean', 'wind_mean', 'wind_period', 'wind_type']
			
tf.compat.v1.enable_eager_execution()

#dicts for ranges, indices
#{key, [min, max, #steps]}
xaxis_dict = {
	'bed_slope_angle' : [0, 30, 9],
	'bed_width' : [1, 100, 9],
	'fuel_clump_size' : [0.5 , 2, 9],
	'fuel_depth' : [0.05, 1, 9],
	'fuel_gap_size' : [0.1, 0.5, 9],
	'fuel_loading' : [0.05, 3, 9],
	'ignition_depth' : [0.1, 4, 9],
	'particle_diameter' : [.0005, .005, 9],
	'particle_moisture' : [2, 35, 9],
	'wind_amplitude_rel_mean' : [0.2, 1, 9],
	'wind_mean' : [0, 10, 9],
	'wind_period' : [1, 5, 9],
}

yaxis_dict = {
	'spread_rate' : [-1, 1, 4],
	'flame_zone_depth' : [-1, 1, 4],
	'flame_length' : [-1, 1, 4]
}

levelv_dict = {
	'bed_slope_angle' : [0, 30, 9],
	'bed_width' : [1, 100, 9],
	'fuel_clump_size' : [0.5 , 2, 9],
	'fuel_depth' : [0.05, 1, 9],
	'fuel_gap_size' : [0.1, 0.5, 9],
	'fuel_loading' : [0.05, 3, 9],
	'ignition_depth' : [0.1, 4, 9],
	'particle_diameter' : [.0005, .005, 9],
	'particle_moisture' : [2, 35, 9],
	'wind_amplitude_rel_mean' : [0.2, 1, 9],
	'wind_mean' : [0, 10, 9],
	'wind_period' : [1, 5, 9]
}

#value int corresponds to key's index in point_list
x_index_dict = {
	'bed_slope_angle' : 0,
	'bed_width' : 1,
	'fuel_clump_size' : 2,
	'fuel_depth' : 3,
	'fuel_gap_size' : 4,
	'fuel_loading' : 5,
	'ignition_depth' : 6,
	'particle_diameter' : 7,
	'particle_moisture' : 8,
	'wind_amplitude_rel_mean' : 9,
	'wind_mean' : 10,
	'wind_period' : 11,
}

y_index_dict = {
	'flame_length' : 0,
	'flame_zone_depth' : 1,
	'spread_rate' : 2
}

axis_label_dict = {
	'bed_slope_angle' : "Bed Slope Angle (degrees)",
	'bed_width' : "Bed Width (m)",
	'fuel_clump_size' : "Fuel Clump Size (m)",
	'fuel_depth' : "Fuel Depth (m)",
	'fuel_gap_size' : "Fuel Gap Size (m)",
	'fuel_loading' : "Fuel Loading (kg/m^3)",
	'ignition_depth' : "Ignition Depth (m)",
	'particle_diameter' : "Particle Diameter (mm)",
	'particle_moisture' : "Particle Moisture (%)",
	'wind_amplitude_rel_mean' : "Wind Amplitude (fraction)",
	'wind_mean' : "Mean Wind Speed (m/s)",
	'wind_period' : "Wind Period (s)",
	'flame_length' : "Flame Length (m)",
	'flame_zone_depth' : "Flame Zone Depth (m)",
	'spread_rate' : "Spread Rate (m/min)"
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
		y_list.append(nested[index])
	return y_list


#conveinent for making multiple data lines for level variable
#gets predictions from create_pred_set and returns list of y_values based on user selection to create single data line
#probably not strictly speaking necessary but here we are
def create_data_line(x_points, name_yaxis, y_dict):
	pred_nums = create_pred_set(x_points)
	y_list = set_y_nums(name_yaxis, pred_nums, y_dict)
	return y_list

#returns nested list of y_list values for graphing lines with level variable
def add_level_vars(name_level, point_list, name_xaxis, name_yaxis, x_dict_index, y_dict_index, x_dict, l_dict):
	levelv_set = []
	y_set = []
	index_x = x_dict_index.get(str(name_xaxis))
	index_l = x_dict_index.get(str(name_level))
	xaxis_points = set_axis(x_dict, name_xaxis, 199) #int is number of datapoints on x-axis
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

#main, does all of the work
def create_app():
	app = Flask(__name__)
	app.secret_key = 'dfjINl!43&Kdfndla*lkn_dl2dsgdHrsd^5H9>46#c*5N'

	try:
		os.makedirs(app.instance_path)
	except OSError:
		pass

	@app.route("/", methods=['GET', 'POST']) #to change url add after "/..."
	#save values in url so can pass link
	@app.route("/<xaxis>&<yaxis>&<levelv>&<mws>&<bsa>&<bw>&<fd>&<fl>&<igd>&<pd>&<pm>&<fcont>&<fcs>&<fgs>&<wc>&<wa>&<wp>")
	def gui_test():
		
		xaxis_points = []
		yaxis_points = []
		
		#get all the user input from selections
		uxaxis = request.form.get('xaxis')
		uyaxis = request.form.get('yaxis')
		ulevelv = request.form.get('levelv')
		ufcont = request.form.get('fcont')
		ufcs = (request.form.get('fcs'))
		ufgs = (request.form.get('fgs'))
		uwc = request.form.get('wc')
		umws = (request.form.get('mws'))
		uwa = (request.form.get('wa'))
		uwp = (request.form.get('wp'))
		ubsa = (request.form.get('bsa'))
		ubw = (request.form.get('bw'))
		ufd = (request.form.get('fd'))
		ufl = (request.form.get('fl'))
		uigd = (request.form.get('igd'))
		upd = (request.form.get('pd'))#to handle change in units
		upm = (request.form.get('pm'))

		point_list=[]
		xaxis_nums=[0,1]
		y_set=[0,1]
		x_range=["wind_mean", 0, 1]
		y_range=["spread_rate", 0, 1]
		radio_points = [0,0]
		axis_labels = [axis_label_dict.get('wind_mean'), axis_label_dict.get('spread_rate'), axis_label_dict.get('particle_moisture')]

		if request.args.get('xaxis') != None: #force init values if no value for xaxis

			#get all the user input from url
			xaxis = request.args.get('xaxis')
			yaxis = request.args.get('yaxis')
			levelv = request.args.get('levelv')
			fcont = request.args.get('fcont')
			fgs = float(request.args.get('fgs'))
			fcs = float(request.args.get('fcs'))
			wc = request.args.get('wc')
			mws = float(request.args.get('mws'))
			wa = float(request.args.get('wa'))
			wp = float(request.args.get('wp'))
			bsa = float(request.args.get('bsa'))
			bw = float(request.args.get('bw'))
			fd = float(request.args.get('fd'))
			fl = float(request.args.get('fl'))
			igd = float(request.args.get('igd'))
			pd = float(request.args.get('pd'))/1000 #to handle change in units
			pm = float(request.args.get('pm'))

		else:
			#init values
			xaxis = 'wind_mean'
			yaxis = 'spread_rate'
			levelv = 'particle_moisture'
			fcont = 'Continuous'
			fcs = 0.0
			fgs = 0.0
			wc = "Constant"
			mws = 4.0
			wa = 0.0
			wp = 0.0
			bsa = 0.0
			bw = 50.0
			fd = 0.2
			fl = 0.2
			igd = 2.0
			pd = 0.002 #to handle change in units
			pm = 10.0

			
		#handles radio button input, allows value to be saved
		if wc == 'Constant':
			wa = 0.0
			wp = 0.0
			w_code = 0.0
			radio_points[0]=0
		else:
			w_code = 1.0
			radio_points[0]=1

		if fcont == 'Continuous':
			fcs = 0.0
			fgs = 0.0
			radio_points[1]=0
		else:
			radio_points[1]=1

		#create list of user input
		point_list.append(bsa)
		point_list.append(bw)
		point_list.append(fcs)
		point_list.append(fd)
		point_list.append(fgs)
		point_list.append(fl)
		point_list.append(igd)
		point_list.append(pd)
		point_list.append(pm)
		point_list.append(wa)
		point_list.append(mws)
		point_list.append(wp)
		point_list.append(w_code)
			
		#used for graph widget
		xaxis_points = set_axis(xaxis_dict, str(xaxis), 199)
		yaxis_points = set_axis(yaxis_dict, str(yaxis), 9)
		levelv_points = set_axis(levelv_dict, str(levelv), 9)

		#obnoxious units fix for particle_diameter
		#don't move to different section of code
		if str(levelv) == "particle_diameter":
			i=0
			while i < len(levelv_points):
				mult_level = levelv_points.copy()
				levelv_points[i] = mult_level[i]*1000
				i+=1

		if str(xaxis) == "particle_diameter":
			i=0
			while i < len(xaxis_points):
				mult_xaxis = xaxis_points.copy()
				xaxis_points[i] = mult_xaxis[i]*1000
				i+=1

		levelv_points.append(levelv)
		x_range = set_range(xaxis_dict, str(xaxis))
		y_range = set_range(yaxis_dict, str(yaxis))

		#init
		y_list = [0, 0, 0, 0]

		axis_labels=[axis_label_dict.get(str(xaxis)), axis_label_dict.get(str(yaxis)), axis_label_dict.get(str(levelv))]

		#get teh predictions/return lists for plotting
		y_set = (add_level_vars(str(levelv), point_list, str(xaxis), str(yaxis), x_index_dict, y_index_dict, xaxis_dict, levelv_dict))
			
		#annoying fix to get particle moistion to save; keep directly above render template or
		#everything will break
		point_list[7]=point_list[7]*1000		
		

		#if user hits calculate
		if request.method == 'POST':
			#udpate page
			return (render_template('base.html', point_list = point_list, xaxis_nums = xaxis_points, yaxis_nums = y_set, x_range = x_range, y_range = y_range, key_labels = levelv_points, radio_points = radio_points, axis_labels = axis_labels) and redirect(url_for('gui_test', xaxis=uxaxis, yaxis=uyaxis, levelv=ulevelv, mws=umws, bsa=ubsa, bw=ubw, fd=ufd, fl=ufl, igd=uigd, pd=upd, pm=upm, fcont=ufcont, fcs=ufcs, fgs=ufgs, wc=uwc, wa=uwa, wp=uwp)))
		
		return render_template('base.html', point_list = point_list, xaxis_nums = xaxis_points, yaxis_nums = y_set, x_range = x_range, y_range = y_range, key_labels = levelv_points, radio_points = radio_points, axis_labels = axis_labels) 

	return app
