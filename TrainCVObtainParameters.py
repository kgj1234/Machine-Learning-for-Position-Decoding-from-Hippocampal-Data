from tkinter import filedialog
from tkinter import *
import numpy as np
import datetime
import pickle as pickle

import RunCrossValidation as RCV
import firingrateapproximations
import velocityfiltering


#Ask for neural activity input. Currently supporting text files
root=Tk()
root.fileName=filedialog.askopenfilename(title="neural activity",filetypes=(("txt files","*.txt"),("All files", "*.*")))
neural_activity=np.transpose(np.loadtxt(root.fileName,delimiter=','))

#Assume there are more time instances than neurons
if neural_activity.shape[0]<neural_activity.shape[1]:
	neuralactivity=np.transpose(neuralactivity)



#Determine if xpos and ypox are combined, then load them
combined=input('Are xpos and ypos combined y/n ')
if combined=='y':
	root.fileName=filedialog.askopenfilename(title="pos",filetypes=(("txt files","*.txt"),("All files", "*.*")))
	pos=np.loadtxt(root.fileName,delimiter=',')
	
	xpos=pos[:,0]
	ypos=pos[:,1]
	
	print(pos.shape)
	
else:
	root.fileName=filedialog.askopenfilename(title="xpos",filetypes=(("txt files","*.txt"),("All files", "*.*")))
	xpos=np.loadtxt(root.fileName,delimiter=',')
	root.fileName=filedialog.askopenfilename(title="ypos",filetypes=(("txt files","*.txt"),("All files", "*.*")))
	ypos=np.loadtxt(root.fileName,delimiter=',')








#Determine the scale, if any that has been applied
scale=float(input('Has the position data been scaled? If so, input the scale, 1 is unscaled: '))




#Use a convolutional neural network?
model_type=input('model_type? NN or CNN: ')



#Keep out a test set  for use after validation, or use all available data. 
sep_test=input('Separate Test Indices from Train Indices? y/n ')
if sep_test.lower()=='y':
	sep_test=True
#Test data is removed from beginning of data set
if sep_test=='y':
	test_size=float(input('Percentage of data to use as test set '))
	sep_test=True
else:
	test_size=None
	sep_test=False
#For use if one has separated out the place cells. This should be an array stored in a text file
neurons=input('Use neuron subset? y/n ')
if neurons.lower()=='y':
	root.fileName=filedialog.askopenfilename(title="neurons",filetypes=(("txt files","*.txt"),("All files", "*.*")))
	neurons=np.loadtxt(root.fileName,delimiter=',')
	neuralactivity=neuralactivity[:,neurons]
else:
	neurons=None

#Determines if data should be used from both sides, or only prior to the prediction time
Left=input('Use neuron activity only  from  time prior to prediction? y/n ')
if Left.lower()=='y':
	Left=True
else:
	Left=False

# Determine whether to perform Cross Validation
cv=input('perform cross validation? y/n ')
if cv=='y':
	cv=True
	
	#Input how much to offset neural data from position by
	offsets=input('Offset the data to account for neuron recording delays (comma separated values): ')
	offsets=offsets.split(',')
	offsets=np.asarray([int(offsets[i]) for i in range(len(offsets))])

	#Determine how much neural data surrounding the data point to use
	length=input('How many time steps of neural data to use from each side in comma separated format: ')
	length=length.split(',')
	length=np.asarray([int(length[i]) for i in range(len(length))])
	
	#Determine how many folds to use in cross validation
	num_folds=int(input('Number of folds? '))

	# Determine how many  neurons to use in cross validation
	num_neurons=input('number of neurons for grid search in comma separated format: ')
	num_neurons=num_neurons.split(',')
	num_neurons=np.asarray([int(num_neurons[i]) for i in range(len(num_neurons))])
	num_layers=input('How many layers in the neural network in comma separated format: ')
	num_layers=num_layers.split(',')
	print(num_neurons)
	num_layers=np.asarray([int(num_layers[i]) for i in range(len(num_layers))])
	print(num_layers)
	#number of iterations per fold
	iterations=int(input('number of  iterations per fold: '))
	plotRes=False
else:
	num_folds=0
	sep_test=True
	test_size=float(input('Percentage of data to use as test set '))
	#Plot results after training, typically no for cross validation
	plotRes=input('Show Plots? y/n ')
	if plotRes=='y':
		plotRes=True
	else:
		plotRes=False
	offsets=[int(input('Offset the data to account for neuron recording delays: '))]
	length=np.array(int(input('How many time steps of neural data to use surrounding the prediction point ')))
	num_neurons=np.array(int(input('number of neurons per fully connected layer? ')))
	iterations=np.array(int(input('number of iterations to train with? ')))
	num_layers=np.array(int(input('How many layers in the neural network: ')))
	
#Apply velocity filtering, this is typically done for hippocampal data
filtering=float(input('velocity filter value, none is 0 '))
if (cv==True and sep_test==True) or cv==False: 
	save_res=input('save results? y/n ')
else:
	save_res=False
	result_name=''
if save_res=='y':
	save_res=True
	result_name=input('Name to save resulting predictions as? ')
else:
	save_res=False
	result_name=''
#Determine desired filter to apply, and input parameters
method=input('neural activity filter, default: none, g: Gaussian, w: Window, a: alpha, trace: approximate trace ')
param_values=[]
if method=='g':
	param_values=input('Input Standard Deviation value. If performing cross validation, comma delimited ')
elif method=='w':
	param_values=input('Input Window Length. If performing cross validation, comma delimited ')
elif method=='a':
	param_values=input('Input alpha value. If performing cross validation, comma delimited ')
elif method =='trace':
	param_values=input('Input trace parameter. If performing cross validation, comma delimited ')
if len(param_values)>0:
	param_values=param_values.split(',')
	param_values=np.asarray([float(param_values[i]) for i in range(len(param_values))])
	if cv==False:
		param_values=np.array(param_values[0])

save_model=input('Save the resulting model? y/n ')
if save_model=='y':
	save_model=True
	model_name=input('Name of folder in which to place model, this will erase an already created folder? ')
else:
	save_model=False
	model_name=''



#For linear track data, we generally remove 5-10 percent of the data from both ends of the track, value should be between 0 and 1
cutoff=float(input('Use if data is from linear track. Input percentage of data to delete from track ends. If not desired, input 0 '))
if cutoff>0:
	tracklength=float('input tracklength: ')
else:
	tracklength=None
	
print(param_values)

RCV.RunCrossValidation(neural_activity,xpos,ypos,scale=scale,cv=cv,length=length,offsets=offsets,neurons=neurons,num_neurons=num_neurons,num_layers=num_layers,param_values=param_values,model_type=model_type,num_folds=num_folds,
						Left=Left,sep_test=sep_test,iterations=iterations,plotRes=plotRes,cutoff=cutoff,tracklength=tracklength,method=method,result_name=result_name,
						test_size=test_size,filtering=filtering,save_model=save_model,model_name=model_name,save_res=save_res)