import tensorflow as tf
import os.path

#prints one prediction from model based on values in line 15

def get_pred():
	#model setup
	model_filename = 'par_model_final/par_model'
	model_path = (str(os.getcwd()) + '/' + model_filename)
	model = tf.keras.models.load_model(model_path)


	tf.compat.v1.enable_eager_execution()

	input_list = [0.5, 5.0, 15.0, 85.0, 1.0, 0.4]
	feature_batch = tf.constant([input_list])
	prediction_as_tensor = model.predict(feature_batch)
	prediction_as_list = (prediction_as_tensor.tolist()[0])
	return prediction_as_list

print(get_pred())
