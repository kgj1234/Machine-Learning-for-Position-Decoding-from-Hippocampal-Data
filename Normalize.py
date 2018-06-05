import numpy as np

def CalcAvgStd(data):

	avg,std=np.zeros((data.shape[1])),np.zeros((data.shape[1]))
	for i in range(0,data.shape[1]):
		avg[i]=np.mean(data[:,i])
		std[i]=np.std(data[:,i])
	return avg,std





def Normalize(data,avg,std):
	for i in range(0,data.shape[1]):
		data[:,i]=(data[:,i]-avg[i])/std[i]
	return data
def Demean(data,avg):
	
	for i in range(data.shape[1]):
		
		data[:,i]=data[:,i]-avg[i]
	
	return data
		
