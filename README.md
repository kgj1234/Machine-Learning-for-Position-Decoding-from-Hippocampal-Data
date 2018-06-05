# Machine-Learning-for-Position-Decoding-from-Hippocampal-Data
This repository contains code designed to predict position of an animal from its hippocampal data. This can be used in experiments, to help describe the effect of stimuli on an animal, or it can be used just to verify the quality of the neural activity.

Currently, linear track and rectangular enclosures are supported. I see no reason why the code would fail for other shaped enclosures, so long as the parameter offset is set to 0.

Neural activity can be formated either as spike trains, in which case the data should be binned, or as processed data (such as the output of CNMF-E). Data should be in a comma delimited text file, and should be of shape n_neurons by n_samples. The code will allow you to select the file, or input the filepath manually.

Position data should be a text file of form n_samples by 2. 


Data and Parameters are inputted using either TrainCVObtainParameters.py or TrainCVObtainParametersManual.py. The first will take you through all of the possible options in an interactive format. The second allows manual input of the parameters, with full descriptions of what each parameter does. In the future, I will create a simplified version of TrainCVObtainParameters.py, which will only ask for the essential parameters.


Results and model parameters are saved in a pickle file at the end of training, with filepath specified by the user.

Note that when selecting a folder to save the resulting models in, if the input matches a folder in the current directory, that folder will be replaced.


Currently, I do not have a program dedicated to extracting predictions from a trained model, though this is on the to do list. predictions can still be extracted after training the model, if this is specified in the TrainCVObtainParameters* code.

Rudimentary testing has been done. Let me know of any issues.
