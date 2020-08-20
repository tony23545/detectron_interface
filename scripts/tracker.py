import numpy as np 
import cv2
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

class Object:
	def __init__(self, pos, idx, dt = 0.05):
		self.pos = pos.reshape((2, 1))
		self.kalman = cv2.KalmanFilter(4,2)
		self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
		self.kalman.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]],np.float32)
		self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.3
		self.kalman.statePost = np.concatenate([pos, np.zeros(2)]).astype(np.float32).reshape(4, 1)
		self.dying = 0
		self.hit = False
		self.id = idx

	def predict(self):
		self.kalman.predict()
		self.pos = self.kalman.statePre[:2]

	def correct(self, measurement):
		self.kalman.correct(measurement)
		self.pos = self.kalman.statePost[:2]

class Tracker:
	def __init__(self):
		self.object_list = []
		self.max_id = 0

	def update(self, detect_list):
		num_of_object = len(self.object_list)
		num_of_detect = len(detect_list)

		for obj in self.object_list:
			obj.predict()

		cost = np.zeros((num_of_object, num_of_detect))

		for i in range(num_of_object):
			for j in range(num_of_detect):
				cost[i, j] = np.linalg.norm(self.object_list[i].pos.reshape(2) - detect_list[j])

		obj_ind, det_ind = linear_sum_assignment(cost)

		for o, d in zip(obj_ind, det_ind):
			self.object_list[o].correct(detect_list[d].astype(np.float32).reshape(2, 1))

		if num_of_object <= num_of_detect: # there are new detection
			self.birth(det_ind, num_of_detect, detect_list)
			#TODO filter out high cost
		else:
			self.death(obj_ind, num_of_object)

	def birth(self, det_ind, num_of_detect, detect_list):
		for det in range(num_of_detect):
			if det not in det_ind:
				self.object_list.append(Object(detect_list[det], self.max_id))
				self.max_id += 1

	def death(self, obj_ind, num_of_object):
		new_object_list = []
		for obj in range(num_of_object):
			if obj not in obj_ind:
				self.object_list[obj].dying += 1
			else:
				self.object_list[obj].dying = 0

			if self.object_list[obj].dying < 3:
				new_object_list.append(self.object_list[obj])
		self.object_list = new_object_list

def main():
	path = np.loadtxt("path.txt").astype(np.float32)
	traj = [np.concatenate([np.zeros((100, 2)), path[:100], np.zeros((200, 2))]),
			np.concatenate([path[100:300], np.zeros((200, 2))]), 
			np.concatenate([np.zeros((50, 2)), path[300:600], np.zeros((50, 2))]),
			path[600:]]
	tracker = Tracker()
	fig = plt.figure()
	color = ['b', 'g', 'y', 'k']
	for t in range(400):
		pose = []
		plt.clf()
		for i in range(4):
			if traj[i][t][0] > 0:
				plt.plot(traj[i][t][0], traj[i][t][1], 'ro')
				pose.append(traj[i][t] + np.random.normal(0, 0.3, (2)))
		tracker.update(pose)

		for obj in tracker.object_list:
			plt.plot(obj.pos[0], obj.pos[1], '+', c = color[obj.id])

		plt.xlim((0, 1500))
		plt.ylim((0, 1500))
		plt.draw()
		plt.pause(0.05)

if __name__ == "__main__":
	main()
