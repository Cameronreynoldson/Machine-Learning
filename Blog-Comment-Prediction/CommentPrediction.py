import numpy as np

def gradient_step(theta,features,learning_rate,targets):
	m = features.size;
	temp_theta = np.array([0]*48)
	prediction = features.dot(np.transpose(theta))
	for i in range(0,47): 
		error_sum = 0
		for j in range(0,109):
			error_sum += (prediction[i]-targets[i])*features[j,i]
		temp_theta_val = (theta[i] - (learning_rate/m)*error_sum)
		temp_theta[i] = temp_theta_val
	return temp_theta

def linear_regression(theta,targets,features,learning_rate,num_iterations):
	for i in range(0,num_iterations):
		theta = gradient_step(theta,features,learning_rate,targets)
	return theta

def run():
	data = np.genfromtxt("example.csv", delimiter=",")
	learning_rate = 0.0001
	theta = np.array([0]*48)
	targets = np.array(data[:,280])
	features = np.array(data[:,0:48])
	features[0] = 1
	num_iterations = 1000
	theta = linear_regression(theta,targets,features,learning_rate,num_iterations)

if __name__ == '__main__':
	run()
