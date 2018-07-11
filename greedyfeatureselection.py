import numpy as np
import datetime
import RunCrossValidation as RCV
import CVExtractParameters
import TrainModelWrapper
import pickle

from contextlib import contextmanager

import sys, os





@contextmanager
def suppress_stdout():

    with open(os.devnull, "w") as devnull:

        old_stdout = sys.stdout

        sys.stdout = devnull

        try:  

            yield

        finally:

            sys.stdout = old_stdout

#Manual Parameter Input

#Path to neural activity. Activity should be in form n_neurons by n_samples, or vice versa, asssuming you have more samples than neurons, comma delimited format. If data is transposed, remove transpose from definition of neural activity
neuron_path="C:/Users/kgj1234/Desktop/Research/MachineLearningPaperData/machinelearningpaperdatafiring.txt"
#Path to position data. Data should be in form n_samples by 2, comma delimited format. 
position_path="C:/Users/kgj1234/Desktop/Research/MachineLearningPaperData/machinelearningpaperdataposition.txt"
neural_activity=np.transpose(np.loadtxt(neuron_path,delimiter=','))
if neural_activity.shape[0]<neural_activity.shape[1]:
	neural_activity=np.transpose(neural_activity)
print(neural_activity.shape)
pos=np.loadtxt(position_path,delimiter=',')
xpos=pos[:,0]
ypos=pos[:,1]

#If position data is scaled, replace scale with the position scale
scale=1





#The amount of neural activity to use from each side of the temporal prediction point, adding more elements to the vector is only valid of cv='y'
length=0

#Use data only from before the temporal prediction point, or allow data to be used from both sides of the point at which position is to be predicted
Left=False

#How much to offset the neural data by. Typically 250 ms,. For the case of my data, this is approximately 4 time steps, adding more elements to the vector is only valid of cv='y'
offset=4



#Number of neurons to use per layer in the neural network, in the case of conv networks, number to use in the fully connected layers, adding more elements to the vector is only valid of cv='y'  
num_neurons=700

#Number of fully connected layers in the network
num_layers=3

#Model Type, currently NN-neural network, and CNN-convolutional neural network, RNN functionality hopefully soon
model_type='NN'

#Separate test set: if true, and cv=True, then cross validation is performed on the train set, the best parameters are found, the model is trained on the test set using these parameters, and tested on the test set,

sep_test=True

#Number of iterations for training. 
iterations=1001

#Whether or not to plot the results, has no effect if cv=True and sep_test=False
plotRes=True


# Indicates whether to remove data from the left and right edges of the track. This may be desirable in the case of linear tracks. If cutoff is not 0, a tracklength is required.
cutoff=0

#Used to cut off data from a linear track
tracklength=None

#Preprocess neural data using various methods: None-no processing, 'g': gaussian, 'a': alpha, 'w': window, 'trace': approximate trace
method='g'

#Parameter values to use if neural data is to be smoothed, for trace: around 1/100, for window and gaussian, around 100, adding more elements to the vector is only valid of cv='y', and if method is not None
param_value=50

#Name results are saved under
result_name='temp results'

#How large of a test set to use, value between 0 and 1
test_size=.2

#Save the model for later use 
save_model=False

#Name of model if saving model, requires  a string even if not saving, This will overwrite a folder if there is already a folder of that name
model_name='./temp'

#Save results- this saves the test set predictions as a txt file
save_res=False

#Name desired result is saved as. This is also the name that the parameter and result pickle file will be saved as. 
result_name='temp'

#If you pass this the filename of a text file containing a vector, it will load the list, and use only the indicated neurons. Use this if you want to filter out certain neurons, or if you have identified place cells
neurons=[i for i in  range(46)]

#Value for velocity filtering. velocity filtering is a typical practice with hippocampal data
filtering=6

Results,Parameters=RCV.RunCrossValidation(neural_activity,xpos,ypos,scale=scale,length=length,offsets=offsets,neurons=neurons,num_neurons=num_neurons,num_layers=num_layers,
						param_values=param_values,model_type=model_type,num_folds=num_folds,
						Left=Left,sep_test=sep_test,iterations=601,cutoff=cutoff,tracklength=tracklength,method=method,
						test_size=test_size,filtering=filtering)
r2=np.mean(np.asarray(Results['length_0offset_4num_neurons_700num_layers3param value50'])[:,0])

current_r2=r2
least_indices=[]
for i in neurons:
	neurons_current=np.setdiff1d(neurons,[i])
	with suppress_stdout():
		Results,Parameters=RCV.RunCrossValidation(neural_activity,xpos,ypos,scale=scale,length=length,offsets=offsets,neurons=neurons,num_neurons=num_neurons,num_layers=num_layers,
						param_values=param_values,model_type=model_type,num_folds=num_folds,
						Left=Left,sep_test=sep_test,iterations=iterations,cutoff=cutoff,tracklength=tracklength,method=method,
						test_size=test_size,filtering=filtering)
		current_r2=np.mean(np.asarray(Results['length_0offset_4num_neurons_700num_layers3param value50'])[:,0])
	if current_r2>r2:
		r2=current_r2
		neurons=neurons_current
		print('neuron removed', str(i))
print(np.setdiff1d([i for i in range(46)],neurons))






while current_r2>.93*r2:			
	least_change=1
	current_index=0
	print(len(neurons))
	for i in neurons:
		print('neuron', str(i))
		neurons_current=np.setdiff1d(neurons,[i])
		with suppress_stdout():
			Results,Parameters=RCV.RunCrossValidation(neural_activity,xpos,ypos,scale=scale,length=length,offsets=offsets,neurons=neurons,num_neurons=num_neurons,num_layers=num_layers,
						param_values=param_values,model_type=model_type,num_folds=num_folds,
						Left=Left,sep_test=sep_test,iterations=iterations,cutoff=cutoff,tracklength=tracklength,method=method,
						test_size=test_size,filtering=filtering)
		if r2-current_r2<least_change:
			least_change=r2-current_r2
			current_index=i
	
	neurons=np.setdiff1d(neurons,[current_index])
	print('neuron removed', current_index)
print(neurons)















