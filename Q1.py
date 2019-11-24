import numpy as np
import math
from random import shuffle
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as pause
from mpl_toolkits.mplot3d import Axes3D
from time import sleep
import matplotlib.animation as animation
from matplotlib import cm
import sys

# DEFAULT PARAMETERS
threshold = 0.0000000000001

def plotPoints(plotOutFunction, fy=[]):
	x = []
	y = []
	
	for i in range(len(xyPoints)):
		x.append(xyPoints[i][0])
		y.append(xyPoints[i][len(xyPoints[0])-1])
	
	plt.title("XY points and predicted function")
	plt.scatter(x,y,color='blue',label='Actual Data')
	if (plotOutFunction == 1):
		plt.plot(x,fy,color='red',label='Curve Fitted')

	plt.xlabel('Input Data')
	plt.ylabel('Output Data')
	plt.legend()
	plt.show()	


def plot3D(xyPoints,plotThetaX, plotThetaY, plotJo, timeGap):
	theta0, theta1 = np.mgrid[0:2:50j, -1:1:50j]
	mesh = np.c_[theta0.flatten(), theta1.flatten()]

	JVal = (np.array([calculateError(xyPoints,point) for point in mesh]).reshape(50,50))

	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.plot_surface(theta0, theta1, JVal, cmap=cm.RdBu_r)
	ax.set_xlabel(r'$\theta_0$', labelpad=10)
	ax.set_ylabel(r'$\theta_1$', labelpad=10)
	ax.set_zlabel(r'$J(\theta)$', labelpad=10)

	plt.show()

	for i in range(len(plotJo)):
		if i % 100 == 0:
			ax.plot([plotThetaX[i]], [plotThetaY[i]], [plotJo[i]], linestyle='-',color='r', marker='o', markersize=3)
			plt.pause(timeGap)

	plt.savefig("Assign1Q1-3DPlot.png")

def plotContours(xyPoints, plotThetaX, plotThetaY, plotJo,title, doMod100 = True):
	theta0, theta1 = np.mgrid[-2:2:50j, -1:2:50j]
	mesh = np.c_[theta0.flatten(), theta1.flatten()]

	JVal = (np.array([calculateError(xyPoints,point) for point in mesh]).reshape(theta0.shape))

	plt.ion()
	plt.title(title)
	plt.contour(theta0, theta1, JVal, 25, colors="k")
	plt.xlabel(r'$\theta_0$', labelpad=10)
	plt.ylabel(r'$\theta_1$', labelpad=10)

	plt.show()

	for i in range(len(plotJo)):
		if doMod100 and i % 100 == 0:
			plt.plot([plotThetaX[i]], [plotThetaY[i]], linestyle='-',color='r', marker='o', markersize=3)
			plt.pause(0.02)
		elif doMod100 == False:
			plt.plot([plotThetaX[i]], [plotThetaY[i]], linestyle='-',color='r', marker='o', markersize=3)
			plt.pause(0.02)

	plt.savefig("Assign1Q1-Contours.png")
	plt.close()

def readPoints(fileX, fileY):
	noOfPoints = 0
	xyPoints = []

	f1 = open(fileX, "r")
	f2 = open(fileY, "r")

	xPoint = f1.read().splitlines()
	yPoint = f2.read().splitlines()

	for i in range(len(xPoint)):
		xy1Point = []
		xy1Point.append(float(xPoint[i]))
		xy1Point.append(1.0)			# This is the value corresponding to theta0 (since theta0 is the intercept)
		xy1Point.append(float(yPoint[i]))

		xyPoints.append(xy1Point)
		noOfPoints = noOfPoints + 1

	f1.close()
	f2.close()

	return xyPoints, noOfPoints

def normalize(xyPoints):
	mean = []
	var = []

	for j in range(len(xyPoints[0])):
		if j == len(xyPoints[0])-2:
			break

		x = []
		for i in range(len(xyPoints)):
			x.append(xyPoints[i][j])

		mean.append(np.mean(x))
		var.append(np.std(x))

		for i in range(len(xyPoints)):
			xyPoints[i][j] = (xyPoints[i][j] - mean[j]) / var[j]

	return xyPoints, mean, var

def unormalize(xyPoints, mean, var, fx=[], isFxAvailable=False):
	for i in range(len(xyPoints[0])):
		if (i == len(xyPoints[0])-2):
			break

		for j in range(len(xyPoints)):
			xyPoints[j][i] = (xyPoints[j][i] * var[i]) + mean[i]

			if isFxAvailable == True:
				fx[j] = (fx[j] * var[i]) + mean[i]

	return xyPoints, fx

def getNewFunction(xyPoints, thetaVec):
	fy = []

	for k in range(len(xyPoints)):
		fy1 = 0.0
		for i in range(len(thetaVec)):
			fy1 = fy1 + thetaVec[i] * xyPoints[k][i]

		fy.append(fy1)

	return fy

def calculateError(xyPoints, thetaVec):
	diffSq = 0.0
	for i in range(len(xyPoints)):
		htheta = 0.0
		for j in range(len(xyPoints[i])-1):
			htheta = htheta + thetaVec[j] * xyPoints[i][j]
			
		diff = htheta - xyPoints[i][len(xyPoints[0])-1]
		diffSq = diffSq + diff * diff

	Jo = diffSq / float(2 * len(xyPoints))
	return Jo

def derivateOfError(thetaVec, xyPoints, featureNo):
	derivJo = 0.0
	for i in range(len(xyPoints)):
		htheta = 0.0
		for j in range(len(xyPoints[i])-1):
			htheta = htheta + thetaVec[j] * xyPoints[i][j]
			
		derivJo = derivJo + (htheta - xyPoints[i][len(xyPoints[0])-1]) * xyPoints[i][featureNo]

	derivJo = derivJo / float(len(xyPoints))
	return derivJo

def updateTheta(xyPoints, thetaVec, alpha):
	tempThetaVec = [0] * (len(xyPoints[0]) - 1)

	# First loop is just to accumulate changed theta. DONOT modify thetaVec in this loop
	for i in range(len(thetaVec)):
		tempThetaVec[i] = thetaVec[i] - alpha * derivateOfError(thetaVec, xyPoints, i)

	# Once all newTheta is calculated, update the thetaVec in this loop
	for i in range(len(thetaVec)):
		thetaVec[i] = tempThetaVec[i]

	return thetaVec

def gradientDescent(xyPoints, noOfPoints, alpha):
	JPvs = 0
	thetaVec = [0] * (len(xyPoints[0]) - 1)		# Last col is y so we will subtract that col in thetaVec and last theta is O0 which we will add a column in theta Vec
	mean = [0] * len(xyPoints[0])
	var = [0] * len(xyPoints[0])

	plotThetaX = []
	plotThetaY = []
	plotJo = []

	while 1:
		thetaVec = updateTheta(xyPoints, thetaVec, alpha)
		JCur = calculateError(xyPoints, thetaVec)

		if JCur == float("inf"):
			break

		if JCur < 1000:
			plotThetaX.append(thetaVec[0])
			plotThetaY.append(thetaVec[1])
			plotJo.append(JCur)

		if abs(JCur-JPvs) <= threshold:
			break							# Converged

		JPvs = JCur

	return thetaVec, plotThetaX, plotThetaY, plotJo

fileX = sys.argv[1] # prints python_script.py
fileY = sys.argv[2] # prints var1
learningRate = float(sys.argv[3]) # prints var2
timeGap = float(sys.argv[4]) # prints var2

xyPoints,noOfPoints = readPoints(fileX, fileY)
xyPoints, mean, var = normalize(xyPoints)

############ Part (a)
alpha = learningRate
thetaVec, plotThetaX, plotThetaY, plotJo = gradientDescent(xyPoints, noOfPoints, alpha)

print ("Learning Rate = ", alpha)
print ("Stopping Criteria = J0 - JPvs <= ", threshold)
print ("Theta Vector = ", thetaVec)

############# Part (b)
fx = getNewFunction(xyPoints, thetaVec)
plotPoints(True, fx)

############# Part (c)
plot3D(xyPoints,plotThetaX, plotThetaY, plotJo, timeGap)

############# Part (d)
plotContours(xyPoints,plotThetaX, plotThetaY, plotJo, "Contours for learning rate = " + str(alpha))

############# Part (e)
alphaL = [0.001, 0.002, 0.003, 0.02, 2.1]
for i in range(len(alphaL)):
	alpha = alphaL[i]
	thetaVec, plotThetaX, plotThetaY, plotJo = gradientDescent(xyPoints, noOfPoints, alpha)
	if i == 4:
		plotContours(xyPoints,plotThetaX, plotThetaY, plotJo, "Contours for learning rate = " + str(alpha),False)
	else:
		plotContours(xyPoints,plotThetaX, plotThetaY, plotJo, "Contours for learning rate = " + str(alpha))
