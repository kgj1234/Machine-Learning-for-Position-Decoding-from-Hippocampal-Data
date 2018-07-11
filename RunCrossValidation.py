
import TrainTest
import TrainTestConvNet
import numpy as np
import datetime
import pickle
import velocityfiltering
import DataPrep
def RunCrossValidation(neuralactivity,xpos,ypos,scale=1,length=[5],offsets=[4],neurons=None,num_neurons=[700],num_layers=[3],param_values=None,model_type='NN',num_folds=5,
						Left=False,sep_test=False,iterations=700,cutoff=0,tracklength=None,method=None,test_size=0,filtering=4,res_folder='./'):
	#Separate out test set if desired
	
	Results_dict={}
		
	current_iteration=1
	for i in length:
		for j in offsets:
			#Offset neural activity, scale, apply neuron subset
			
			Neuron,position=DataPrep.prepDataTrain(neuralactivity,xpos,ypos,scale,neurons,j)
			if sep_test==True:
		
				test_indices=np.arange(0,int(test_size*Neuron.shape[0]))
				train_indices=np.setdiff1d(np.arange(0,Neuron.shape[0]),test_indices)
			else:
				train_indices=np.arange(0,Neuron.shape[0])
			for k in num_neurons:
				for m in num_layers:
					if param_values is not None:
						for l in param_values:
							print('cross validation',str(current_iteration)+ ' of '+str(len(length)*len(offsets)*len(num_neurons)*len(param_values)*len(num_layers)))
							#key results will be saved under in pickle file
							key='length_'+str(i)+'offset_'+str(j)+'num_neurons_'+str(k)+'num_layers'+str(m)+'param value'+str(l)
							current_result=[]
							print('length', i)
							print('offset', j)
							print('num_neurons', k)
							print('num_layers', m)
							print('parameter value', l)
							
	
							
							for q in range(num_folds):
								print('fold'+ str(q+1)+' of '+str(num_folds))
						
								
								train_Neuron,train_position,test_Neuron,test_position=DataPrep.Separate_Normalize_Batch_CV_Train_Set(Neuron,position,train_indices,num_folds,q,i,Left_Only=Left,
																																	method=method,param_values=l,normalize=False,filtering=filtering)
							
								train_Neuron,train_position,test_Neuron,test_position=DataPrep.cutoff_ends(cutoff,train_Neuron,train_position,test_Neuron,test_position,tracklength=tracklength)	
								train_Neuron,train_position,test_Neuron,test_position=np.asarray(train_Neuron),np.asarray(train_position),np.asarray(test_Neuron),np.asarray(test_position)
								#Train Models
								if model_type=='NN':
							
									r2,rmse,predicted,actual=TrainTest.train_model(train_Neuron,train_position,test_Neuron,
																			test_position,iterations,n_neurons=k,n_layers=m)
								if model_type=='CNN':
									r2,rmse,predicted,actual=TrainTestConvNet.train_convnet(train_Neuron,train_position,
																			test_Neuron,test_position,iterations,n_neurons=k,n_layers=m)
								current_result.append([r2,rmse])
								
								
							Results_dict[key]=current_result
							current_iteration+=1
			
					else:
							
						print(str(current_iteration)+ ' of '+str(len(length)*len(offsets)*len(num_neurons)*len(num_layers)))
						key='length_'+str(i)+'offset_'+str(j)+'num_neurons_'+str(k)+'num_layers'+str(m)
						
						current_result=[]
						for q in range(num_folds):
								
						
							
							train_Neuron,train_position,test_Neuron,test_position=DataPrep.Separate_Normalize_Batch_CV_Train_Set(Neuron,position,train_indices,num_folds,q,i,Left_Only=Left,
																																	method=None,param_values=None,normalize=False,filtering=filtering)
							train_Neuron,train_position,test_Neuron,test_position=DataPrep.cutoff_ends(cutoff,train_Neuron,train_position,test_Neuron,test_position,tracklength=tracklength)				
							#Train Models
							if model_type=='NN':
								r2,rmse,predicted,actual=TrainTest.train_model(np.asarray(train_Neuron),np.asarray(train_position),np.asarray(test_Neuron),np.asarray(test_position),iterations,n_neurons=k,n_layers=m)
							if model_type=='CNN':
								r2,rmse,predicted,actual=TrainTestConvNet.train_convnet(np.asarray(train_Neuron),np.asarray(train_position),np.asarray(test_Neuron),np.asarray(test_position),iterations,n_neurons=k,n_layers=m)
							current_result.append([r2,rmse])
								
								
						Results_dict[key]=current_result
						current_iteration+=1

	
	
	
				
	Parameters_dict={}

	Parameters_dict['Left']=Left
	Parameters_dict['num_folds']=num_folds
	Parameters_dict['model_type']=model_type
	
	Parameters_dict['sep_test']=sep_test
	Parameters_dict['neurons']=neurons
	Parameters_dict['test_size']=test_size
	Parameters_dict['params']=param_values
	Parameters_dict['velocity filter']=filtering
	Parameters_dict['method']=method
	Parameters_dict['num_neurons']=num_neurons
	
	Parameters_dict['offsets']=offsets
	Parameters_dict['num_layers']=num_layers
	Parameters_dict['cutoff']=cutoff
	Parameters_dict['length']=length
	Parameters_dict['tracklength']=tracklength
	Parameters_dict['scale']=scale
	return Results_dict,Parameters_dict



