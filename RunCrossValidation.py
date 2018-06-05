import ProcessForTraining as PFT
import numpy as np
import datetime
import pickle
import velocityfiltering
def RunCrossValidation(neuralactivity,xpos,ypos,scale=1,cv=False,length=[5],offsets=[4],neurons=None,num_neurons=[700],num_layers=[3],param_values=[],model_type='NN',num_folds=[0],
						Left=False,sep_test=False,iterations=1000,plotRes=False,cutoff=0,tracklength=None,method=None,result_name='newresult',test_size=0,filtering=4,save_model=True,model_name='./newmodel',save_res=True):
	
	if filtering>0:
		filtered=velocityfiltering.velocity_filter(xpos,ypos,filtering)
		neuralactivity=neuralactivity[filtered,:]
		xpos=xpos[filtered]
		ypos=ypos[filtered]
	if neurons is not None:
		neurons=np.loadtxt(root.fileName,delimiter=',')	
		neuralactivity=neuralactivity[:,neurons]

	Results_dict={}

	if cv==True:
		
		current_iteration=1
		for i in range(len(length)):
			for j in range(len(offsets)):
				for k in range(len(num_neurons)):
					for m in range(len(num_layers)):
						if len(param_values)>0:
							for l in range(len(param_values)):
								print(str(current_iteration)+ 'of'+str(len(length)*len(offsets)*len(num_neurons)*len(param_values)*len(num_layers)))
						
								key='length_'+str(length[i])+'offset_'+str(offsets[j])+'num_neurons_'+str(num_neurons[k])+'num_layers'+str(num_layers[m])+'param value'+str(param_values[l])
								current_result=PFT.Process_and_Input(neuralactivity,xpos,ypos,scale=scale,offset=offsets[j],length=length[i],model_type=model_type,
									num_folds=num_folds,num_neurons=num_neurons[k],Left=Left,sep_test=sep_test,
									iterations=iterations,plotRes=plotRes,cutoff=cutoff,tracklength=tracklength,method=method,
									param_values=param_values[l],result_name=result_name,test_size=test_size,num_layers=num_layers[m],save_model=save_model,model_name=model_name,save_res=save_res)

								Results_dict[key]=current_result
								current_iteration+=1
						else:
							
							print(str(current_iteration)+ 'of'+str(len(length)*len(offsets)*len(num_neurons)*len(num_layers)))
							key='length_'+str(length[i])+'offset_'+str(offsets[j])+'num_neurons_'+str(num_neurons[k])+'num_layers'+str(num_layers[m])
							current_result=PFT.Process_and_Input(neuralactivity,xpos,ypos,scale=scale,
								offset=offsets[j],length=length[i],model_type=model_type,
								num_folds=num_folds,num_neurons=num_neurons[k],Left=Left,sep_test=sep_test,
								iterations=iterations,plotRes=plotRes,result_name=result_name,cutoff=cutoff,tracklength=tracklength,
								method=None,param_values=None,test_size=test_size,num_layers=num_layers[m],save_model=save_model,model_name=model_name,save_res=save_res)
							Results_dict[key]=current_result
							current_iteration+=1

	else:
		if len(param_values)>0:
			key='length_'+str(length[0])+'offset_'+str(offsets[0])+'num_neurons_'+str(num_neurons[0])+'num_layers'+str(num_layers[0])+'param value'+str(param_values[0])
			current_result,avg,std,pos_avg=PFT.Process_and_Input(neuralactivity,xpos,ypos,
				scale=scale,offset=offsets[0],length=length[0],model_type=model_type,num_folds=0,
				num_neurons=num_neurons[0],Left=Left,sep_test=sep_test,iterations=iterations,plotRes=plotRes,
				cutoff=cutoff,tracklength=tracklength,result_name=result_name,method=method,
				param_values=param_values,test_size=test_size,save_res=save_res,num_layers=num_layers[0],save_model=save_model,model_name=model_name)

			Results_dict[key]=current_result
		else:
			key='length_'+str(length[0])+'offset_'+str(offsets[0])+'num_neurons_'+str(num_neurons[0])+'num_layers'+str(num_layers[0])
			current_result,avg,std,pos_avg=PFT.Process_and_Input(neuralactivity,xpos,ypos,
				scale=scale,offset=offsets[0],length=length[0],model_type=model_type,num_folds=0,
				num_neurons=num_neurons[0],Left=Left,sep_test=sep_test,iterations=iterations,plotRes=plotRes,
				cutoff=cutoff,tracklength=tracklength,result_name=result_name,method=None,param_values=None,
				test_size=test_size,save_res=save_res,num_layers=num_layers[0],save_model=save_model,model_name=model_name)	
			Results_dict[key]=current_result
	Compiled_Results={}
	for key in Results_dict:
		
		Compiled_Results[key]=np.mean(np.asarray(Results_dict[key]),axis=0)
	r2=0
	final_key=None
	for key in Results_dict:
		if Compiled_Results[key][0]>r2:
			r2=Compiled_Results[key][0]
			final_key=key
	print('Best Parameters:', final_key)
	if cv==True and sep_test==True:
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
		
		if len(param_values)>0:
			
			index2=final_key.find('param value')
			num_layers=int(final_key[index1+10:index2])
			param_values=float(final_key[index2+12:])
			
			current_result,avg,std,pos_avg=PFT.Process_and_Input(neuralactivity,xpos,ypos,
			scale=scale,offset=offset,length=length,model_type=model_type,num_folds=0,
			num_neurons=num_neurons,Left=Left,sep_test=sep_test,iterations=iterations,plotRes=True,
			cutoff=cutoff,tracklength=tracklength,method=method,param_values=param_values,
			save_res=save_res,num_layers=num_layers,result_name=result_name)
			print(current_result)
		else:
			
			
			num_layers=int(final_key[index1+10:])
			current_result,avg,std,pos_avg=PFT.Process_and_Input(neuralactivity,xpos,ypos,
			scale=scale,offset=offset,length=length,model_type=model_type,num_folds=0,
			num_neurons=num_neurons,Left=Left,sep_test=sep_test,iterations=iterations,plotRes=True,
			cutoff=cutoff,tracklength=tracklength,method=None,param_values=None,
			save_res=save_res,num_layers=num_layers,result_name=result_name)
			print(current_result)
	elif cv==True and sep_test==False:
		avg,std,pos_avg=None,None,None
			
				


	Results_dict['Left']=Left
	Results_dict['num_folds']=num_folds
	Results_dict['model_type']=model_type
	
	Results_dict['sep_test']=sep_test
	Results_dict['neurons']=neurons
	Results_dict['test_size']=test_size
	Results_dict['params']=param_values
	Results_dict['velocity filter']=filtering
	Results_dict['method']=method
	Results_dict['num_neurons']=num_neurons
	Results_dict['Neuron_avg']=avg
	Results_dict['Neuron_std']=std
	Results_dict['position_avg']=pos_avg
	Results_dict['offsets']=offsets
	Results_dict['num_layers']=num_layers
	now=datetime.datetime.now()




	if save_model==True:

		savetitle=model_name+"CrossValResults"+str(now.strftime("%Y-%m-%d"))+result_name+'.p'
	else:
		savetitle="CrossValResults"+str(now.strftime("%Y-%m-%d"))+result_name+'.p'
	pickle.dump(Results_dict,open(savetitle,"wb"))




