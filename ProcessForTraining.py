import numpy as np
from matplotlib import pyplot as plt
import DataPrep
import TrainTest
import TrainTestConvNet
import Normalize
import time
import TrainTestRNN
def Process_and_Input(Neuron,xpos,ypos,neurons=None,num_folds=0,iterations=1000,
		length=10,num_neurons=1000,cutoff=0,offset=5,scale=1,Left=False,model_type='NN',
		tracklength=None,
		sep_test=False,plotRes=False,method=None,param_values=None,save_res=True,result_name='current',num_layers=0,test_size=.2,save_model=False,model_name='./newmodel'):
	
	
	
	Neuron,position=DataPrep.prepDataTrain(Neuron,xpos,ypos,scale,offset,length)
	
	
	if sep_test==True:
		
		test_indices=np.arange(0,int(test_size*Neuron.shape[0]))
		train_indices=np.setdiff1d(np.arange(0,Neuron.shape[0]),test_indices)
	else:
		train_indices=np.arange(0,Neuron.shape[0])
	
		
		
		
	print(Neuron.shape)	
	Results=[]
	if num_folds>0:
		
		for i in range(num_folds):
			train_Neuron,train_position,test_Neuron,test_position=DataPrep.Separate_Normalize_Batch_CV_Train_Set(Neuron,position,train_indices,num_folds,i,length,Left_Only=False,method=method,param_values=param_values)
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
					print('removing data may have caused a significant portion of the test set to be lost')



			
			if model_type=='NN':
				r2,rmse,predicted,actual=TrainTest.train_model(np.asarray(train_Neuron),np.asarray(train_position),np.asarray(test_Neuron),np.asarray(test_position),iterations,num_neurons,folder=model_name,save_model=save_model)
				Results.append([r2,rmse])
				
				
			if model_type=='conv':
				r2,rmse,predicted,actual=TrainTestConvNet.train_convnet(np.asarray(train_Neuron),np.asarray(train_position),np.asarray(test_Neuron),np.asarray(test_position),iterations,num_neurons,folder=model_name,save_model=save_model)
				Results.append([r2,rmse])
	elif sep_test==True:
		
		train_Neuron,train_position,test_Neuron,test_position,avg,std,pos_avg=DataPrep.Separate_Normalize_Batch_Test_Train_Set(Neuron,position,train_indices,test_indices,length,method=method,param_values=param_values,Left_Only=Left)
		if cutoff>0:
				if tracklength is None:
					tracklength=100
				include_train=[i for i in range(len(train_position)) if train_position[i][0]>cutoff*100 and train_position[i][0]<(100-cutoff*100)]
				train_Neuron=train_Neuron[include_train]
				train_position=train_position[include_train]
				include_test=[i for i in range(len(test_position)) if test_position[i][0]>cutoff*100 and test_position[i][0]<(100-cutoff*100)]
				test_Neuron=test_Neuron[include_test]
				test_position=test_position[include_test]
				if len(position_test)<.17*len(position_train):
					print('removing data caused a significant portion of the test set to be lost')
		if model_type=='NN':
			r2,rmse,predicted,actual=TrainTest.train_model(np.asarray(train_Neuron),np.asarray(train_position),np.asarray(test_Neuron),np.asarray(test_position),iterations,num_neurons,folder=model_name,save_model=save_model)
			Results.append([r2,rmse])
				
				
		if model_type=='conv':
			r2,rmse,predicted,actual=TrainTestConvNet.train_convnet(np.asarray(train_Neuron),np.asarray(train_position),np.asarray(test_Neuron),np.asarray(test_position),iterations,num_neurons,folder=model_name,save_model=save_model)
														
			Results.append([r2,rmse])	
		#RNN functionality will be added later.
		#if model_type=='RNN':
		#	TrainTestRNN.TrainRNN(np.asarray(train_Neuron),np.asarray(train_position),np.asarray(test_Neuron),np.asarray(test_position))
	else:
		print('when training, it is advised to separate out a test set')
		
	if plotRes==True:	
		x=np.linspace(0,len(actual),len(actual))
		plt.plot(x[0:10000],actual[0:10000,0],color='blue')
		plt.plot(x[0:10000],predicted[0:10000,0],color='red')
		plt.show()
		time.sleep(1)
		plt.close()
		if save_res==True:
			np.savetxt(result_name+'pred.txt',predicted,delimiter=',')
			np.savetxt(result_name+'actual.txt',actual,delimiter=',')
	if num_folds==0:
		return Results,avg,std,pos_avg
	return Results

