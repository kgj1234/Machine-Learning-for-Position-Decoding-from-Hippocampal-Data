# Gaussian Convolution
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse import random
import numpy as np
from matplotlib import pyplot as plt
def gaussianConvolution(neural_activity,standard_dev):
	#print(neural_activity.shape[1])
	for  i in range(0,neural_activity.shape[1]):
		neural_activity[:,i]=gaussian_filter(neural_activity[:,i],sigma=standard_dev)
	return neural_activity


def windowMethod(neural_activity,window_length):
	padding=np.zeros((window_length,neural_activity.shape[1]))
	neural_activity=np.vstack((padding,neural_activity,padding))

	for i in range(neural_activity.shape[1]):
		total=np.zeros((neural_activity.shape[0]-2*window_length))
		for j in range(-window_length,window_length+1):
		
		
			total=total+neural_activity[window_length-j:neural_activity.shape[0]-window_length-j,i]
		
		neural_activity[window_length:-window_length,i]=total
	neural_activity=neural_activity*1/(2*window_length)
	return neural_activity[window_length+1:-window_length-1,:]
	
def alpha_function(neural_activity,alpha,window_length=100):
	padding=np.zeros((window_length,neural_activity.shape[1]))
	neural_activity=np.vstack((padding,neural_activity,padding))

	for i in range(neural_activity.shape[1]):
		total=np.zeros((neural_activity.shape[0]-2*window_length))
		for j in range(0,window_length):
		
		
			total=total+(alpha**(2))*abs(j)*np.exp(-alpha*abs(j))*neural_activity[window_length-j:neural_activity.shape[0]-window_length-j,i]
		
		neural_activity[window_length:-window_length,i]=total
	#neural_activity[window_length:2*window_length]*=2
	
	#neural_activity[-window_length:]*=2
	neural_activity=neural_activity
	return neural_activity[window_length:-window_length,:]
def approximateTraceFromSpiking(firing_data,alpha):
	y=np.zeros((len(firing_data)))
	x=np.linspace(0,len(firing_data),len(firing_data))
	expo=np.exp(-alpha*x)

	length=len(firing_data)
	for i in range(len(firing_data)):
		if firing_data[i]>0:
		
			y[i:i+min([100,length-i])]=y[i:i+min([100,length-i])]+firing_data[i]*expo[0:min([100,length-i])]
			
	return y	
def ApproxTrace(firing_matrix,alpha=.056):
	y=np.zeros((firing_matrix.shape))
	for i in range(len(firing_matrix[0,:])):

		y[:,i]=approximateTraceFromSpiking(firing_matrix[:,i],alpha=alpha)

	return y	



#Testing
"""
np.random.seed(seed=42)
x=np.linspace(0,1000,1000)
y=random(1000,1,density=.1)
print(y)
y=y.toarray()

Y1=y.copy()
#print(Y1)

y1=gaussianConvolution(Y1,10)
Y1=y.copy()
y2=windowMethod(Y1,10)
y3=alpha_function(Y1,.1)
print(Y1)
y4=ApproxTrace(Y1)

fig,ax=plt.subplots(5,1)
ax[0].plot(x,y,color='black',label='ground truth')
ax[1].plot(x,y1,color='blue',label='convolution')

#print(x[1:-1].shape)
#print(y2.shape)
ax[2].plot(x[1:-1],y2,color='red',label='window')
ax[3].plot(x[0:-1],y3,color='green',label='alpha')
ax[4].plot(x,y4,color='purple',label='approx trace')
ax[0].set_ylabel('Spikes')
ax[1].set_ylabel('gaussian filter')
ax[2].set_ylabel('window')
ax[3].set_ylabel('alpha')
ax[4].set_ylabel('appox trace')


plt.show()
"""