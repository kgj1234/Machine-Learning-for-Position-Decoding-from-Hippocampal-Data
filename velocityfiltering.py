import numpy as np
def velocity_filter(xpos,ypos,filter_vel,dt=.0256):
	xvel=(xpos[1:]-xpos[0:-1])/dt
	yvel=(ypos[1:]-ypos[0:-1])/dt

	vel=np.sqrt((xvel)**2+(yvel)**2)
	vel=np.hstack((vel[0],vel))
	filtered=[i for i in range(len(vel)) if vel[i]>filter_vel]
	return filtered
	

	