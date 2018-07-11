import TestModelWrapper
import numpy as np

import pickle



#Run Test Model Wrapper 

#Parameters Path
Parameters_path='C:/Users/kgj1234/Desktop/Machine-Learning-for-Position-Decoding-from-Hippocampal-Data-master/gaussianmodelNN/Parameters2018-06-22gaussian_result.p'
#Neural Activity path
neuron_path="C:/Users/kgj1234/Desktop/Research/MachineLearningPaperData/machinelearningpaperdatafiring.txt"
#position path
position_path="C:/Users/kgj1234/Desktop/Research/MachineLearningPaperData/machinelearningpaperdataposition.txt"

#model path
model_path='C:/Users/kgj1234/Desktop/Machine-Learning-for-Position-Decoding-from-Hippocampal-Data-master/gaussianmodelNN/'


neural_activity=np.transpose(np.loadtxt(neuron_path,delimiter=','))
if neural_activity.shape[0]<neural_activity.shape[1]:
	neural_activity=np.transpose(neural_activity)

pos=np.loadtxt(position_path,delimiter=',')
xpos=pos[:,0]
ypos=pos[:,1]

with open(Parameters_path, 'rb') as f:
    Parameters = pickle.load(f)

test_indices=np.arange(int(.2*neural_activity.shape[0]),neural_activity.shape[0])
neural_activity=neural_activity[test_indices,:]
xpos=xpos[test_indices]
ypos=ypos[test_indices]


differences=[]


control_r2,control_rmse,pred,actual,gradients=TestModelWrapper.TestModelWrapper(neural_activity,xpos,ypos,scale=1,length=10,offset=4,neurons=None,param_value=None,model_type='NN',
						Left=False,plot_res=False,cutoff=0,tracklength=None,method=None,result_name='testresult',filtering=4,model_name=model_path,save_res=True,
						Parameters=Parameters,test_size=1,return_grad=False,zero=None)
						
for i in range(46):
		r2,rmse,pred,actual,gradients=TestModelWrapper.TestModelWrapper(neural_activity,xpos,ypos,scale=1,length=10,offset=4,neurons=None,param_value=None,model_type='NN',
						Left=False,plot_res=False,cutoff=0,tracklength=None,method=None,result_name='testresult',filtering=4,model_name=model_path,save_res=True,
						Parameters=Parameters,test_size=1,return_grad=False,zero=[i])
		differences.append(control_r2-r2)
print(differences)
						