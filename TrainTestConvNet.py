import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.externals import joblib
from random import shuffle
import TrainTest
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def conv_layer(data,fmaps,ksize,stride,pad,batch_normalize=True):
	if batch_normalize==False:
		return tf.layers.conv2d(data,fmaps,ksize,strides=stride,padding=pad,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(.05))
	else:
		return tf.nn.relu(tf.layers.batch_normalization(tf.layers.conv2d(data,fmaps,ksize,strides=stride,padding=pad,kernel_initializer=tf.contrib.layers.xavier_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(.05))))
def pool_layer(data,ksize,strides,padding):
	return tf.layers.average_pooling2d(data,pool_size=ksize,strides=strides,padding=padding)
def combined_layer(data,neurons,training):
	

	return dropout_layer(hidden_layer(data,neurons,training),training)
def hidden_layer(data,neurons,training):
	
	return tf.layers.batch_normalization(tf.layers.dense(data,neurons,tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(.05)),training=training)
def dropout_layer(data,training):
	return tf.layers.dropout(data,.5,training=training)
def train_convnet(X_train_batches,y_train_batches,X_test_batches,y_test_batches,iterations,n_neurons,folder='./newmodel',n_layers=0,save_model=False):
	tf.reset_default_graph()
	
	
	
	print('number of training samples: ',X_train_batches.shape[0])
	print('number of testing samples: ',X_test_batches.shape[0])
	print('number of features: (',X_train_batches.shape[1],',',X_train_batches.shape[2],')')
	
	

	training=tf.placeholder(tf.bool,None,name='Training')
	learning_rate=tf.placeholder(tf.float32,None)

	conv1_fmaps=32
	conv1_ksize=3
	conv1_stride=1
	conv1_pad="SAME"

	conv2_fmaps=64
	conv2_ksize=3
	conv2_stride=2
	conv2_pad="SAME"


	



	n_fcl=n_neurons
	n_outputs=2
	X=tf.placeholder(tf.float32,[None,X_train_batches[0].shape[0],X_train_batches[0].shape[1]])
	
	X_reshaped=tf.reshape(X,[tf.shape(X)[0],X_train_batches[0].shape[0],X_train_batches[0].shape[1],1])

	y=tf.placeholder(tf.float32,shape=[None,2], name='y')

	conv1=conv_layer(X_reshaped,conv1_fmaps,conv1_ksize,conv1_stride,conv1_pad)
	
	dropout=dropout_layer(conv1,training)
	conv2=conv_layer(dropout,conv2_fmaps,conv2_ksize,conv2_stride,conv2_pad)


	pool=tf.nn.max_pool(conv2,[1,2,2,1],[1,2,2,1],"VALID")
	

					
	dropout1=tf.layers.dropout(pool,rate=.5,training=training,name='dropout')					
	size=int(dropout1.shape[2])
	size1=int(dropout1.shape[3])
	size2=int(dropout1.shape[1])
	
	dropout_flat=tf.reshape(dropout1,shape=[-1,size*size1*size2])
	
	
	if n_layers==0:
		prediction=tf.layers.dense(dropout_flat,2)
	else:
		nn=combined_layer(dropout_flat,n_neurons,training)
		for i in range(n_layers-1):
			nn=combined_layer(dropout_flat,n_neurons,training)
		prediction=tf.layers.dense(nn,2)



	loss=tf.reduce_mean(tf.square(prediction-y))
	update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
		training_op=optimizer.minimize(loss)
	total_error = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
	unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y, prediction)))
	R_squared = tf.subtract(1., tf.div(unexplained_error,total_error))
	init=tf.global_variables_initializer()
	Saver=tf.train.Saver()
	n_iterations=iterations
	i=0


	rate=.01
	
	loss=tf.reduce_mean(tf.square(prediction-y))
	update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
		training_op=optimizer.minimize(loss)
	total_error = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
	unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y, prediction)))
	R_squared = tf.subtract(1., tf.div(unexplained_error,total_error))
	

	init=tf.global_variables_initializer()
	Saver=tf.train.Saver()
	n_iterations=iterations
	
	
	
	
	#Construct builder to save Model
	if save_model==True:
		#Overwrite folder or request new folder to save models in 
		while True:
			if os.path.isdir(folder):
				del_dir=input('Delete Directory and Replace with New Models? y/n ')
				if del_dir=='y':
					shutil.rmtree(folder)
					break
				else:
					folder=input('input a new folder name: ')
			else:
				break
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
	batch_size=2000
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess=tf.Session(config=config)
	
	
	max_r2=-10
	count=0
	
	
	
	with sess.as_default():
		init.run()
		
		
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
				test_mse=[]
				r2_test=[]
				#Note this will not calculate mse or r2 on test samples with some indices
				for i in range(int(len(y_test_batches)/batch_size)):
					test_mse.append(loss.eval(feed_dict={X:X_test_batches[i*batch_size:(i+1)*batch_size],y:y_test_batches[i*batch_size:(i+1)*batch_size],training:False}))
					r2_test.append(R_squared.eval(feed_dict={X:X_test_batches[i*batch_size:(i+1)*batch_size],y:y_test_batches[i*batch_size:(i+1)*batch_size],training:False}))
				test_mse=np.mean(test_mse)
				r2_test=np.mean(r2_test)
				
				if iteration%200==0:
					try:
						print('iteration',iteration,'Train RMSE',np.sqrt(train_mse),'Test RMSE',np.sqrt(test_mse),'Test R2', np.mean(r2_test))
					except:
						pass
				if r2_test>max_r2:
					max_r2=r2_test
					count=0
				else:
					count+=1
				if count>=10:
					count=0
					rate=rate/2
					
		if save_model==True:
			builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING],
							signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:prediction_signature})
			builder.save()	
					
		try:		
			print('final r2',r2_test,'final rmse', np.sqrt(test_mse))
		except:
			pass
		
	
		pred=[]
		for i in range(int(X_test_batches.shape[0]/batch_size)+1):
			current_pred=sess.run(prediction,feed_dict={X:X_test_batches[batch_size*i:batch_size*(i+1)],training:False})
			pred.append(current_pred)
		pred=np.vstack(pred)
		
		
	sess.close()
	return r2_test,np.sqrt(test_mse),pred,np.array(y_test_batches)





	

def test_convnet(X_test_batches,y_test_batches,folder='./',return_grad=False):
	tf.reset_default_graph()
	tf.logging.set_verbosity(tf.logging.ERROR)
	print('number of testing samples: ',X_test_batches.shape[0])
	print('number of features: (',X_test_batches.shape[1],',',X_test_batches.shape[2],')')
	
	
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
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
	
	
	
	
	
		
		
		
	batch_size=2000
	test_mse=[]
	r2_test=[]
	for i in range(int(X_test_batches.shape[0]/batch_size)):
		test_mse.append(loss.eval(feed_dict={X:X_test_batches[batch_size*i:batch_size*(i+1)],y:y_test_batches[batch_size*i:batch_size*(i+1)],training:False},session=sess))
		r2_test.append(R_squared.eval(feed_dict={X:X_test_batches[batch_size*i:batch_size*(i+1)],y:y_test_batches[batch_size*i:batch_size*(i+1)],training:False},session=sess))

	test_mse=np.mean(test_mse)
	r2_test=np.mean(r2_test)
	
	pred=[]
	for i in range(int(X_test_batches.shape[0]/batch_size)+1):
		current_pred=sess.run(prediction,feed_dict={X:X_test_batches[batch_size*i:batch_size*(i+1)],training:False})
		pred.append(current_pred)

	pred=np.vstack(pred)
	
	if return_grad==True:
		
		x_grad=tf.gradients(prediction[:,0],X)
	
		y_grad=tf.gradients(prediction[:,1],X)
		x_gradients=[]
		y_gradients=[]
		
		
		for j in range(int(X_test_batches.shape[0]/batch_size)+1):
		
			x_grad_eval=sess.run(x_grad,feed_dict={X:X_test_batches[batch_size*j:batch_size*(j+1),:,:],training:False})
	
			x_gradients.append(x_grad_eval[0])
			y_grad_eval=sess.run(y_grad,feed_dict={X:X_test_batches[batch_size*j:batch_size*(j+1),:,:],training:False})
			y_gradients.append(y_grad_eval[0])
		x_gradients=np.concatenate(x_gradients)
		y_gradients=np.concatenate(y_gradients)
		#average over magnitude
		x_gradients=np.sqrt(np.sum(np.square(x_gradients),0)/x_gradients.shape[0])
		y_gradients=np.sqrt(np.sum(np.square(y_gradients),0)/y_gradients.shape[0])
	
		gradients=np.concatenate((x_gradients.reshape((1,x_gradients.shape[0],x_gradients.shape[1])),y_gradients.reshape((1,y_gradients.shape[0],y_gradients.shape[1]))))
	
			
        
	else:
		gradients=None
        
		
	return r2_test,np.sqrt(test_mse),pred,y_test_batches,gradients
		


	
