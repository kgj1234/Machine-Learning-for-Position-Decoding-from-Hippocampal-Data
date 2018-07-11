import TestModelWrapper
import numpy as np
from matplotlib import pyplot as plt
import pickle



#Run Test Model Wrapper 

#Parameters Path
Parameters_path='C:/Users/kgj1234/Desktop/Machine-Learning-for-Position-Decoding-from-Hippocampal-Data-master/SpikeNN/Parameters2018-06-27spike_results.p'
#Neural Activity path
neuron_path="C:/Users/kgj1234/Desktop/Research/MachineLearningPaperData/machinelearningpaperdatafiring.txt"
#position path
position_path="C:/Users/kgj1234/Desktop/Research/MachineLearningPaperData/machinelearningpaperdataposition.txt"

#model path
model_path='C:/Users/kgj1234/Desktop/Machine-Learning-for-Position-Decoding-from-Hippocampal-Data-master/SpikeNN/'


neural_activity=np.transpose(np.loadtxt(neuron_path,delimiter=','))
if neural_activity.shape[0]<neural_activity.shape[1]:
	neural_activity=np.transpose(neural_activity)

pos=np.loadtxt(position_path,delimiter=',')
xpos=pos[:,0]
ypos=pos[:,1]

with open(Parameters_path, 'rb') as f:
    Parameters = pickle.load(f)


test_indices=np.arange(int(neural_activity.shape[0]*.2),neural_activity.shape[0])

neural_activity=neural_activity[test_indices,:]




xpos=xpos[test_indices]
ypos=ypos[test_indices]
print(xpos.shape)
r2,rmse,pred,actual,gradients=TestModelWrapper.TestModelWrapper(neural_activity,xpos,ypos,scale=1,length=10,offset=4,neurons=None,param_value=None,model_type='NN',
						Left=False,plot_res=False,cutoff=0,tracklength=None,method=None,result_name='testresult',filtering=4,model_name=model_path,save_res=False,Parameters=Parameters,test_size=1,return_grad=True)
						

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
np.savetxt('sensitiveindices.txt',sorted_indices,delimiter=',')

