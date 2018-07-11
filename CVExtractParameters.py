#Determine Best Parameters from Cross Validation, and Extract said parameters for training
import numpy as np
def best_key(Results_dict):
	Compiled_Results={}
	for key in Results_dict:
		
		Compiled_Results[key]=np.mean(np.asarray(Results_dict[key]),axis=0)
	r2=0
	final_key=None
	for key in Results_dict:
		if Compiled_Results[key][0]>r2:
			r2=Compiled_Results[key][0]
			final_key=key
	return final_key
	
def extract_param(final_key,method):
	
	index1=final_key.find('length_')
	index2=final_key.find('offset_')
	length=int(final_key[index1+7:index2])
	index1=index2
	index2=final_key.find('num_neurons_')
	offset=int(final_key[index1+7:index2])
	index1=index2
	index2=final_key.find('num_layers')
	num_neurons=int(final_key[index1+12:index2])
	index1=index2
		
	if method is not None:
			
		index2=final_key.find('param value')
		
		num_layers=int(final_key[index1+10:index2])
		param_value=float(final_key[index2+11:])
		
		key='length_10offset_4num_neurons_700num_layers3param value100'
		
	else:
		num_layers=int(final_key[index1+10:])
		param_value=None
	
	return length,offset,num_neurons,num_layers,param_value
