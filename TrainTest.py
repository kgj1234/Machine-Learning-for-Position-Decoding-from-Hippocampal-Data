import numpy as np
import os

import tensorflow as tf
import shutil

from random import shuffle



def hidden_layer(data,neurons,training):
	
	return tf.layers.batch_normalization(tf.layers.dense(data,neurons,tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(.05)),training=training)
def dropout_layer(data,training):
	return tf.layers.dropout(data,.5,training=training)
def combined_layer(data,neurons,training):
	

	return dropout_layer(hidden_layer(data,neurons,training),training)

def train_model(X_train_batches,y_train_batches,X_test_batches,y_test_batches,iterations,n_neurons,n_layers=3,folder='./newmodel',save_model=True):
	tf.reset_default_graph()
	
	
	


	
	n_outputs=2
	training=tf.placeholder(tf.bool,None,name='Training')
	learning_rate=tf.placeholder(tf.float32,None)
	
	X=tf.placeholder(tf.float32,[None,X_train_batches[0].shape[0],X_train_batches[0].shape[1]])
	X_reshaped=tf.reshape(X,[-1,X_train_batches[0].shape[0]*X_train_batches[0].shape[1]])
	print(X_reshaped.shape)
	y=tf.placeholder(tf.float32,[None,n_outputs])
	if n_layers==0:
		prediction=tf.layers.dense(X_reshaped,2)
	else:
		nn=combined_layer(X_reshaped,n_neurons,training)
		for i in range(n_layers-1):
			nn=combined_layer(nn,n_neurons,training)
		prediction=tf.layers.dense(nn,2)
	


	
	# if capped==True:
		
		# maximum=tf.constant([100.-100*float(cutoff),50.])
		# minimum=tf.constant([100*float(cutoff),0.])
		# prediction=tf.minimum(prediction,maximum)
		# prediction=tf.maximum(prediction,minimum)
	loss=tf.reduce_mean(tf.square(prediction-y))
	update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
		training_op=optimizer.minimize(loss)
	
	
	total_errorx = tf.reduce_sum(tf.square(tf.subtract(y[:,0], tf.reduce_mean(y[:,0]))))
	unexplained_errorx = tf.reduce_sum(tf.square(tf.subtract(y[:,0], prediction[:,0])))
	R_squaredx = tf.subtract(1., tf.div(unexplained_errorx,total_errorx))
		
	total_errory = tf.reduce_sum(tf.square(tf.subtract(y[:,1], tf.reduce_mean(y[:,1]))))
	unexplained_errory = tf.reduce_sum(tf.square(tf.subtract(y[:,1], prediction[:,1])))
	R_squaredy = tf.subtract(1., tf.div(unexplained_errory,total_errory))

	init=tf.global_variables_initializer()
	Saver=tf.train.Saver()
	n_iterations=iterations
	
	i=0
	min_rmse=500*500
	max_r2=[-10,-10]
	count=0
	rate=.1
	
	#Construct builder to save Model
	
	if save_model==True:
		if os.path.isdir(folder):
			
			shutil.rmtree(folder)
		builder=tf.saved_model.builder.SavedModelBuilder(folder)
		tensor_info_X=tf.saved_model.utils.build_tensor_info(X)
		tensor_info_y=tf.saved_model.utils.build_tensor_info(y)
		tensor_info_pred=tf.saved_model.utils.build_tensor_info(prediction)
		tensor_info_training=tf.saved_model.utils.build_tensor_info(training)
		prediction_signature=(tf.saved_model.signature_def_utils.build_signature_def(
							inputs={"X":tensor_info_X,"y":tensor_info_y,"training":tensor_info_training},
							outputs={"pred":tensor_info_pred},
							method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
		
	

	shufflearray=list(range(len(X_train_batches)))
	shuffle(shufflearray)
	X_train_batches=X_train_batches[shufflearray]
	y_train_batches=y_train_batches[shufflearray]
	batch_size=min([len(X_train_batches),120])
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess=tf.Session(config=config)
	with sess.as_default():
		init.run()
		if save_model==True:
			builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING],
					signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:prediction_signature})
		
		
		for iteration in range(n_iterations):
			
			X_batch,y_batch=X_train_batches[i*batch_size:(i+1)*batch_size],y_train_batches[i*batch_size:(i+1)*batch_size]
			i+=1
			if len(X_batch)==0:
				i=0
				X_batch,y_batch=X_train_batches[i*batch_size:(i+1)*batch_size],y_train_batches[i*batch_size:(i+1)*batch_size]
				i+=1
			
			
			sess.run(training_op,feed_dict={X:X_batch,y:y_batch,training:True,learning_rate:rate})
			
			if iteration%20==0:
				
				
				train_mse=loss.eval(feed_dict={X:X_batch,y:y_batch,training:False,learning_rate:rate})
				
				X_batch,y_batch=X_test_batches,y_test_batches
			
				test_mse=loss.eval(feed_dict={X:X_batch,y:y_batch,training:False,learning_rate:rate})
				r2_testx=R_squaredx.eval(feed_dict={X:X_batch,y:y_batch,training:False,learning_rate:rate})
				r2_testy=R_squaredy.eval(feed_dict={X:X_batch,y:y_batch,training:False,learning_rate:rate})
				r2_test=[r2_testx,r2_testy]
				if iteration %200==0:
					try:
						print(iteration,np.sqrt(train_mse),np.sqrt(test_mse))
					except:
						pass
				if np.mean(r2_test)>np.mean(max_r2):
					max_r2=r2_test
					count=0
					if save_model==True:
						if os.path.isdir(folder):
			
							shutil.rmtree(folder)
						builder.save()
					
				if min_rmse>np.sqrt(test_mse):
					min_rmse=np.sqrt(test_mse)
					
					count=0
				else:
					count+=1
				if count>10:
					rate=rate/2
					
					
					
		try:		
			print(max_r2,min_rmse,np.mean(max_r2))
		except:
			pass
		pred=sess.run(prediction,feed_dict={X:X_test_batches,training:False})
		
	sess.close()
	return np.mean(max_r2),min_rmse,pred,np.array(y_test_batches)





	

def test_model(X_test_batches,y_test_batches,folder='./'):
	tf.reset_default_graph()
	

	
	
	

	
	sess=tf.Session(config=config)
	model=tf.saved_model.loader.load(sess,["serve"],folder)
	signature=model.signature_def
	signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
	input_X=signature[signature_key].inputs['X'].name
	input_y=signature[signature_key].inputs['y'].name
	input_training=signature[signature_key].inputs['training'].name
	output_prediction=signature[signature_key].outputs['pred'].name
	
	X=sess.graph.get_tensor_by_name(input_X)
	y=sess.graph.get_tensor_by_name(input_y)
	training=sess.graph.get_tensor_by_name(input_training)
	prediction=sess.graph.get_tensor_by_name(output_prediction)
	
	
	#When I get around to learning it, these operations can be retrieved from the saved model. This should work for now.
	total_error = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
	unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y, prediction)))
	R_squared = tf.subtract(1., tf.div(unexplained_error,total_error))
	
	loss=tf.reduce_mean(tf.square(prediction-y))
	
	
	
	
	
		
		
		
		
			
	test_mse=loss.eval(feed_dict={X:X_test_batches,y:y_test_batches,training:False})
	r2_test=R_squared.eval(feed_dict={X:X__test_batches,y:y_test_batches,training:False})
	pred=sess.run(prediction,feed_dict={X:X_test_batches,training:False})	
		
		
	return pred,y_test_batches	
		


