import numpy as np
class StateKalmanFilter(object):

    def __init__(self, A, B, C, x, P, Q, R):
        self.A = A  # Process dynamics
        self.B = B  # Control dynamics
        self.C = C  # Measurement dynamics
        self.current_state_estimate = x  # Current state estimate
        self.current_prob_estimate = P  # Current probability of state estimate
        self.Q = Q  # Process covariance
        self.R = R  # Measurement covariance  #$ error in the observation
        self.H = np.identity(P.shape[0])


    def current_state(self):
        return self.current_state_estimate

    def step(self, control_input, measurement):
        # Prediction step
        self.predicted_state_estimate = np.dot(self.A,self.current_state_estimate) + np.multiply(self.B,control_input)
        self.predicted_prob_estimate = np.dot(np.dot(self.A,self.current_prob_estimate),self.A.T)+ self.Q #state covariance matrix
        
        # Observation step
        self.innovation_covariance = np.dot(np.dot(self.H,self.predicted_prob_estimate),self.H.T) + self.R
        self.innovation = np.dot(self.C,measurement) - np.dot(self.H,self.predicted_state_estimate)


        # Update step
        self.kalman_gain = np.dot(self.predicted_prob_estimate,self.H.T)/ self.innovation_covariance#$ self.kalman_gain
        
        self.current_state_estimate = self.predicted_state_estimate + np.dot(np.diag(np.diag(self.kalman_gain)), self.innovation) #$ current estimate #$ self.kalman_gain

        self.current_prob_estimate = np.dot((1 - np.dot(self.kalman_gain,self.C.T)), self.predicted_prob_estimate) #$ updated error estimate #$self.kalman_gain
        self.current_prob_estimate=np.diag(np.diag(self.current_prob_estimate))