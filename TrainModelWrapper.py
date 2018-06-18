import TrainTest
import TrainTestConvNet
import numpy as np
import datetime
import pickle
import velocityfiltering
import DataPrep
from matplotlib import pyplot as plt
def TrainModelWrapper(neuralactivity,xpos,ypos,scale=1,length=10,offset=4,neurons=None,num_neurons=700,num_layers=3,param_value=None,model_type='NN',
						Left=False,iterations=1000,plot_res=True,cutoff=0,tracklength=None,method=None,result_name='newresult',test_size=.2,filtering=4,save_model=True,model_name='./newmodel',save_res=True):
	
	Parameters={}
	Neuron,position=DataPrep.prepDataTrain(neuralactivity,xpos,ypos,scale,filtering,neurons,offset)
	test_indices=np.arange(0,int(test_size*Neuron.shape[0]))
	train_indices=np.setdiff1d(np.arange(0,Neuron.shape[0]),test_indices)
	
	if param_value is not None:
		key='length_'+str(length)+'offset_'+str(offset)+'num_neurons_'+str(num_neurons)+'num_layers'+str(num_layers)+'param value'+str(param_value)
		print(key)
		train_Neuron,train_position,test_Neuron,test_position,avg,stdev,pos_avg=DataPrep.Separate_Normalize_Batch_Test_Train_Set(Neuron,position,train_indices,test_indices,length,Left_Only=Left,method=method,param_values=param_value)
		train_Neuron,train_position,test_Neuron,test_position=DataPrep.cutoff_ends(cutoff,train_Neuron,train_position,test_Neuron,test_position,tracklength=tracklength)
		
		if model_type=='NN':
			r2,rmse,predicted,actual=TrainTest.train_model(np.asarray(train_Neuron),np.asarray(train_position),np.asarray(test_Neuron),
																			np.asarray(test_position),iterations,num_neurons,n_layers=num_layers,folder=model_name,save_model=save_model)
		if model_type=='CNN':
			r2,rmse,predicted,actual=TrainTestConvNet.train_convnet(np.asarray(train_Neuron),np.asarray(train_position),
																			np.asarray(test_Neuron),np.asarray(test_position),iterations,num_neurons,n_layers=num_layers,folder=model_name,save_model=save_model)

		Parameters[key]=[r2,rmse]
	else:
		key='length_'+str(length)+'offset_'+str(offset)+'num_neurons_'+str(num_neurons)+'num_layers'+str(num_layers)
		
		train_Neuron,train_position,test_Neuron,test_position,avg,stdev,pos_avg=DataPrep.Separate_Normalize_Batch_Test_Train_Set(Neuron,position,train_indices,test_indices,length,Left_Only=Left,method=method,param_values=param_value)
		train_Neuron,train_position,test_Neuron,test_position=DataPrep.cutoff_ends(cutoff,train_Neuron,train_position,test_Neuron,test_position,tracklength=tracklength)
		
		if model_type=='NN':
			r2,rmse,predicted,actual=TrainTest.train_model(np.asarray(train_Neuron),np.asarray(train_position),np.asarray(test_Neuron),
																			np.asarray(test_position),iterations,num_neurons=n_neurons,n_layers=num_layers,folder=model_name,save_model=save_model)
		if model_type=='CNN':
			r2,rmse,predicted,actual=TrainTestConvNet.train_convnet(np.asarray(train_Neuron),np.asarray(train_position),
																			np.asarray(test_Neuron),np.asarray(test_position),iterations,n_neurons=num_neurons,n_layers=num_layers,folder=model_name,save_model=save_model)
		Parameters[key]=[r2,rmse]
	if save_res==True:
		if save_model==True:
			np.savetxt(model_name+'/'+result_name+'pred.txt',predicted,delimiter=',')
			np.savetxt(model_name+'/'+result_name+'actual.txt',actual,delimiter=',')
		else:
			np.savetxt(result_name+'pred.txt',predicted,delimiter=',')
			np.savetxt(result_name+'actual.txt',actual,delimiter=',')
	Parameters['Left']=Left

	Parameters['model_type']=model_type

	
	Parameters['neurons']=neurons
	Parameters['test_size']=test_size
	Parameters['params']=param_value
	Parameters['velocity filter']=filtering
	Parameters['method']=method
	Parameters['num_neurons']=num_neurons
	Parameters['Neuron_avg']=avg
	Parameters['Neuron_std']=stdev
	Parameters['position_avg']=pos_avg
	Parameters['offsets']=offset
	Parameters['length']=length
	Parameters['cutoff']=cutoff
	Parameters['tracklength']=tracklength
	Parameters['scale']=scale
	now=datetime.datetime.now()
	if save_model==True:

		savetitle=model_name+"/Parameters"+str(now.strftime("%Y-%m-%d"))+result_name+'.p'
	else:
		savetitle="Parameters"+str(now.strftime("%Y-%m-%d"))+result_name+'.p'
	pickle.dump(Parameters,open(savetitle,"wb"))
	if plot_res==True:
		x=np.linspace(0,len(actual),len(actual))
		plt.plot(x[0:10000],actual[0:10000,0],color='blue')
		plt.plot(x[0:10000],predicted[0:10000,0],color='red')
		plt.show()

	
	
	
	