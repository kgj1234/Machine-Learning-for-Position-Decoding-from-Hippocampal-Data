import firingrateapproximations as fra
import TrainTest
import numpy as np
import Normalize
import velocityfiltering
from matplotlib import pyplot as plt
def prepDataTrain(Neuron,xpos,ypos,scale,neurons=None,offset=4):
	if neurons is not None:
		if type(neurons)==str:
			neurons=np.loadtxt(neurons,delimiter=',',dtype='int')	
		
		Neuron=Neuron[:,neurons]
		
	
	xpos=xpos*scale
	ypos=ypos*scale
	
		
	
	pos=np.hstack((xpos.reshape((-1,1)),ypos.reshape((-1,1))))
		
	Neuron=Neuron[(offset+1):,:]
	pos=pos[:-offset-1,:]

	return Neuron,pos


def batches(X,y,length,Left_Only=False):
	X_batches=[]
	y_batches=[]
	for i in range(X.shape[0]-2*length-2):
		if Left_Only==False:
			X_batches.append(X[i:i+2*length+1,:])
			y_batches.append(y[i+length,:])
		else:
			X_batches.append(X[i:i+length+1,:])
			y_batches.append(y[i+length,:])
	return 	X_batches,y_batches
#Fix this so it returns the parameters for the normalization for later use


		




def Separate_Normalize_Batch_CV_Train_Set(Neuron,position,train_indices,num_folds,i,length,Left_Only=False,method=None,param_values=None,normalize=False,filtering=0):
	print('number of folds', num_folds)
		
	cv_test_indices=train_indices[i*int(len(train_indices)/num_folds):(i+1)*int(len(train_indices)/num_folds)]
	if i>0:
		
		cv_train_indices1=train_indices[0:i*int(len(train_indices)/num_folds)]
		cv_train_indices2=train_indices[(i+1)*int(len(train_indices)/num_folds):]
		
		
		Neuron1=Neuron[cv_train_indices1,:]
		Neuron2=Neuron[cv_train_indices2,:]
		test_Neuron=Neuron[cv_test_indices,:]
		
		
		if method is not None:
			print('processing method',method)
			if method=='g':
				
				test_Neuron=fra.gaussianConvolution(test_Neuron,param_values)
				Neuron1=fra.gaussianConvolution(Neuron1,param_values)
				Neuron2=fra.gaussianConvolution(Neuron2,param_values)
			if method=='w':
				test_Neuron=fra.windowMethod(test_Neuron,param_values)
				Neuron1=fra.windowMethod(Neuron1,param_values)

				Neuron2=fra.windowMethod(Neuron2,param_values)

			if method=='a':
				test_Neuron=fra.alpha_function(test_Neuron,param_values)
				Neuron1=fra.alpha_function(Neuron1,param_values)
				Neuron2=fra.alpha_function(Neuron2,param_values)

			if method=='trace':
				test_Neuron=fra.ApproxTrace(test_Neuron,param_values)
				Neuron1=fra.ApproxTrace(Neuron1,param_values)
				Neuron2=fra.ApproxTrace(Neuron2,param_values)

		train_Neuron=np.vstack((Neuron1,Neuron2))
		
		
		avg,std=Normalize.CalcAvgStd(train_Neuron)
		test_Neuron=Normalize.Normalize(test_Neuron,avg,std)
		Neuron1=Normalize.Normalize(Neuron1,avg,std)
		Neuron2=Normalize.Normalize(Neuron2,avg,std)
		
		
		
		#Comment out next 6 lines to remove normalization
		if normalize==True:		
			
			avg_pos=np.mean(np.vstack((position[cv_train_indices1,:],position[cv_train_indices2])),0)
			position1=Normalize.Demean(position[cv_train_indices1,:],avg_pos)
			position2=Normalize.Demean(position[cv_train_indices2,:],avg_pos)
			test_position=Normalize.Demean(position[cv_test_indices,:],avg_pos)
			
		else:
			
		


		
			position1=position[cv_train_indices1,:]
			position2=position[cv_train_indices2,:]		




	
		
		
	
			test_position=position[cv_test_indices,:]
		
			
		Neuron1,position1=batches(Neuron1,position1,length,Left_Only)
		Neuron2,position2=batches(Neuron2,position2,length,Left_Only)
		test_Neuron,test_position=batches(test_Neuron,test_position,length,Left_Only)
		train_Neuron=Neuron1+Neuron2
		train_position=position1+position2
		
		
		




		
	else:
		
		cv_train_indices=train_indices[int(len(train_indices)/num_folds):]
		
			

		
		train_Neuron=Neuron[cv_train_indices,:]
		train_position=position[cv_train_indices,:]
		
		
		
		test_position=position[cv_test_indices,:]
		test_Neuron=Neuron[cv_test_indices,:]				

		
	
		if method is not None:
		
			if method=='g':
				print('processing method', method)
				test_Neuron=fra.gaussianConvolution(test_Neuron,param_values)
				train_Neuron=fra.gaussianConvolution(train_Neuron,param_values)
			if method=='w':
				test_Neuron=fra.windowMethod(test_Neuron,param_values)
				train_Neuron=fra.windowMethod(train_Neuron,param_values)

			if method=='a':
				test_Neuron=fra.alpha_function(test_Neuron,param_values)
				train_Neuron=fra.alpha_function(train_Neuron,param_values)

			if method=='trace':
				test_Neuron=fra.ApproxTrace(test_Neuron,param_values)
				train_Neuron=fra.ApproxTrace(train_Neuron,param_values)
		avg,std=Normalize.CalcAvgStd(train_Neuron)
		
		train_Neuron=Normalize.Normalize(train_Neuron,avg,std)
		test_Neuron=Normalize.Normalize(test_Neuron,avg,std)
		if normalize==True:
			
			avg_pos=np.mean(position[cv_train_indices,:],0)
			train_position=Normalize.Demean(position[cv_train_indices,:],avg_pos)
			test_position=Normalize.Demean(position[cv_test_indices,:],avg_pos)
			
			
	
		train_Neuron,train_position=batches(train_Neuron,train_position,length,Left_Only)
		test_Neuron,test_position=batches(test_Neuron,test_position,length,Left_Only)
	
	
	
	
	test_position,train_position=np.asarray(test_position),np.asarray(train_position)
	test_Neuron,train_Neuron=np.asarray(test_Neuron),np.asarray(train_Neuron)
	
	if filtering>0:
		filtered=velocityfiltering.velocity_filter(train_position[:,0],train_position[:,1],filtering)

		train_Neuron=train_Neuron[filtered,:,:]
		train_position=train_position[filtered,:]
		
		filtered=velocityfiltering.velocity_filter(test_position[:,0],test_position[:,1],filtering)
		test_Neuron=test_Neuron[filtered,:,:]
		test_position=test_position[filtered,:]
		
		
	
	return train_Neuron,train_position,test_Neuron,test_position
def cutoff_ends(cutoff,train_Neuron,train_position,test_Neuron,test_position,tracklength=None):
	if cutoff>0:
		if tracklength is None:
			tracklength=100
			include_train=[i for i in range(len(train_position)) if train_position[i][0]>cutoff*100 and train_position[i][0]<(100-cutoff*100)]
			train_Neuron=train_Neuron[include_train]
			train_position=train_position[include_train]
			include_test=[i for i in range(len(test_position)) if test_position[i][0]>cutoff*100 and test_position[i][0]<(100-cutoff*100)]
			test_Neuron=test_Neuron[include_test]
			test_position=test_position[include_test]
			if len(position_test)<(test_size-.03)*len(position_train):
				cont_training=('removing data caused test size to reduce to '+str(len(position_test)/(len(position_test)+len(position_train)))+' of the total. Continue training? y/n ')
				if cont_training=='n':
					raise Exception('Insufficient test set')
	return train_Neuron,train_position,test_Neuron,test_position
def Separate_Normalize_Batch_Test_Train_Set(Neuron,position,train_indices,test_indices,length,Left_Only=False,method=None,param_values=None,normalize=False,avg=None,std=None,pos_avg=None,zero=None,filtering=0):	
	
	
	#avg_pos=np.mean(position[train_indices,:],0)
	#train_position=Normalize.Demean(position[train_indices,:],avg_pos)
	print(position.shape)
	
	
	train_position=position[train_indices,:]
	train_Neuron=Neuron[train_indices,:]
	test_Neuron=Neuron[test_indices,:]
	
	test_position=position[test_indices,:]
	print(test_position.shape)
	if method is not None:
		print('processing method', method)
		if method=='g':
				
			test_Neuron=fra.gaussianConvolution(test_Neuron,param_values)
			train_Neuron=fra.gaussianConvolution(train_Neuron,param_values)
		if method=='w':
			test_Neuron=fra.windowMethod(test_Neuron,param_values)
			train_Neuron=fra.windowMethod(train_Neuron,param_values)

		if method=='a':
			test_Neuron=fra.alpha_function(test_Neuron,param_values)
			train_Neuron=fra.alpha_function(train_Neuron,param_values)

		if method=='trace':
			test_Neuron=fra.ApproxTrace(test_Neuron,param_values)
			train_Neuron=fra.ApproxTrace(train_Neuron,param_values)
	
	
	
	if avg is None:
		avg,std=Normalize.CalcAvgStd(Neuron[train_indices,:])
	train_Neuron=Normalize.Normalize(train_Neuron,avg,std)
	test_Neuron=Normalize.Normalize(test_Neuron,avg,std)
	
	if normalize==True:
		if avg_pos is None:
			avg_pos=avg_pos=np.mean(position[train_indices,:],0)
		train_position=Normalize.Demean(position[train_indices,:],avg_pos)
		test_position=Normalize.Demean(position[test_indices,:],avg_pos)
	else:
		avg_pos=None
	
	

	if zero is not None:
		test_Neuron[:,zero]=0
	train_Neuron,train_position=batches(train_Neuron,train_position,length,Left_Only)
	test_Neuron,test_position=batches(test_Neuron,test_position,length,Left_Only)
	test_Neuron,train_Neuron=np.asarray(test_Neuron),np.asarray(train_Neuron)
	test_position,train_position=np.asarray(test_position),np.asarray(train_position)
	print(train_position.shape)
	if filtering>0:
		if train_position.shape[0]>0:
			filtered=velocityfiltering.velocity_filter(train_position[:,0],train_position[:,1],filtering)
			train_Neuron=train_Neuron[filtered,:,:]
			train_position=train_position[filtered,:]
		
		filtered=velocityfiltering.velocity_filter(test_position[:,0],test_position[:,1],filtering)
		test_Neuron=test_Neuron[filtered,:,:]
		test_position=test_position[filtered,:]
	
	
	
	return train_Neuron,train_position,test_Neuron,test_position,avg,std,avg_pos
		
		
		
		
