import TrainTest
import TrainTestConvNet
import numpy as np
import datetime
import pickle
import velocityfiltering
import DataPrep
from matplotlib import pyplot as plt
def TestModelWrapper(neuralactivity,xpos,ypos,scale=1,length=10,offset=4,neurons=None,param_value=None,model_type='NN',
						Left=False,plot_res=True,cutoff=0,tracklength=None,method=None,result_name='newresult',filtering=4,model_name='./newmodel',save_res=True,Parameters=None,test_size=1):
	
	if Parameters is not None:
		Left=Parameters['Left']

		model_type=Parameters['model_type']

	
		neurons=Parameters['neurons']
		
		param_value=Parameters['params']
		filtering=Parameters['velocity filter']
		method=Parameters['method']
	
		avg=Parameters['Neuron_avg']
		stdev=Parameters['Neuron_std']
		pos_avg=Parameters['position_avg']
		
		length=Parameters['length']
		cutoff=Parameters['cutoff']
		tracklength=Parameters['tracklength']
		scale=Parameters['scale']
	else:
		avg,pos_avg,stdev=None,None,None
	Neuron,position=DataPrep.prepDataTrain(neuralactivity,xpos,ypos,scale,filtering,neurons,offset)
	test_indices=np.arange(0,int(test_size*Neuron.shape[0]))
	if test_size<1:
		train_indices=np.setdiff1d(np.arange(0,Neuron.shape[0]),test_indices)
	else:
		train_indices=np.arange(2)
	
	if param_value is not None:
		
		
		train_Neuron,train_position,test_Neuron,test_position,avg,std,avg_pos=DataPrep.Separate_Normalize_Batch_Test_Train_Set(Neuron,position,train_indices,test_indices,length,Left_Only=Left,method=method,param_values=param_value,avg=avg,std=stdev,pos_avg=pos_avg)
		train_Neuron,train_position,test_Neuron,test_position=DataPrep.cutoff_ends(cutoff,train_Neuron,train_position,test_Neuron,test_position,tracklength=tracklength)
		
		if model_type=='NN':
			print(2)
			r2,rmse,predicted,actual=TrainTest.test_model(np.asarray(test_Neuron),np.asarray(test_position),model_name)
		if model_type=='CNN':
			r2,rmse,predicted,actual=TrainTestConvNet.test_convnet(np.asarray(test_Neuron),np.asarray(test_position),model_name)
		print(r2,rmse)
		
	else:

		
		train_Neuron,train_position,test_Neuron,test_position,avg,std,avg_pos=DataPrep.Separate_Normalize_Batch_Test_Train_Set(Neuron,position,train_indices,test_indices,length,Left_Only=Left,method=method,param_values=param_value,avg=avg,std=stdev,pos_avg=pos_avg)
		train_Neuron,train_position,test_Neuron,test_position=DataPrep.cutoff_ends(cutoff,train_Neuron,train_position,test_Neuron,test_position,tracklength=tracklength)
		
		if model_type=='NN':
			r2,rmse,predicted,actual=TrainTest.test_model(np.asarray(test_Neuron),np.asarray(test_position),model_name)
		if model_type=='CNN':
			r2,rmse,predicted,actual=TrainTestConvNet.test_convnet(np.asarray(test_Neuron),np.asarray(test_position),model_name)
		print(r2,rmse)
	print(predicted.shape)
	if save_res==True:
	
		np.savetxt(model_name+'/'+result_name+'pred.txt',predicted,delimiter=',')
		np.savetxt(model_name+'/'+result_name+'actual.txt',actual,delimiter=',')
		

	now=datetime.datetime.now()
	
	if plot_res==True:
		x=np.linspace(0,len(actual),len(actual))
		plt.plot(x[0:10000],actual[0:10000,0],color='blue')
		plt.plot(x[0:10000],predicted[0:10000,0],color='red')
		plt.show()
