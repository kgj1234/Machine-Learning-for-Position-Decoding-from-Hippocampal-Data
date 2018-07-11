#Noisy sensitivity
import TestModelWrapper
import numpy as np
import datetime
import RunCrossValidation as RCV
import CVExtractParameters
import TrainModelWrapper
import pickle
from matplotlib import pyplot as plt

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
plotRes=False


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
save_model=True

#Name of model if saving model, requires  a string even if not saving, This will overwrite a folder if there is already a folder of that name
model_name='./NoisyNN'

#Save results- this saves the test set predictions as a txt file
save_res=False

#Name desired result is saved as. This is also the name that the parameter and result pickle file will be saved as. 
result_name='temp'

#If you pass this the filename of a text file containing a vector, it will load the list, and use only the indicated neurons. Use this if you want to filter out certain neurons, or if you have identified place cells
neurons=None

#Value for velocity filtering. velocity filtering is a typical practice with hippocampal data
filtering=6

np.random.seed(seed=31)

noisy_activity=np.zeros((neural_activity.shape[0],23))
for i in range(23):
	rand=np.random.uniform(.06,.5)
	noisy_activity[:,i]=np.random.poisson(rand,neural_activity.shape[0])
neural_activity=np.hstack((neural_activity,noisy_activity))





r2,rmse=TrainModelWrapper.TrainModelWrapper(neural_activity,xpos,ypos,scale=1,length=length,offset=offset,neurons=neurons,num_neurons=num_neurons,num_layers=num_layers,
									param_value=param_value,model_type=model_type,
									Left=False,iterations=600,plot_res=False,cutoff=cutoff,tracklength=tracklength,method=method,result_name=result_name,test_size=test_size,
									filtering=filtering,save_model=save_model,model_name=model_name,save_res=save_res)

test_indices=np.arange(int(.2*neural_activity.shape[0]),neural_activity.shape[0])

neural_activity=neural_activity[test_indices,:]
xpos=xpos[test_indices]
ypos=ypos[test_indices]


with open('./NoisyNN/Parameters2018-06-22temp.p', 'rb') as f:
    Parameters = pickle.load(f)
	
r2,rmse,pred,actual,gradients=TestModelWrapper.TestModelWrapper(neural_activity,xpos,ypos,scale=1,length=5,offset=4,neurons=None,param_value=50,model_type='NN',
						Left=False,plot_res=False,cutoff=0,tracklength=None,method='g',result_name='temp',filtering=4,model_name=model_name,save_res=False,Parameters=Parameters,test_size=1,return_grad=True)
						
						
fig,ax=plt.subplots(2,1)
ax[0].imshow(gradients[0],cmap='hot',interpolation='nearest')
ax[1].imshow(gradients[1],cmap='hot',interpolation='nearest')
plt.show()
sensitivities=np.max(gradients,0)
#print(sensitivities)
plt.imshow(sensitivities,cmap='hot',interpolation='nearest')
plt.show()


sensitivities=np.mean(sensitivities,0)

sorted_indices=np.argsort(sensitivities)[::-1]
print(sorted_indices)
print(sensitivities[sorted_indices])
np.savetxt('sensitiveindicesnoisy.txt',sorted_indices,delimiter=',')
												