import numpy as np
import math
import pandas as pd
from random import shuffle
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as pause
from mpl_toolkits.mplot3d import Axes3D
from time import sleep
import matplotlib.animation as animation
import sys

# DEFAULT PARAMETERS
threshold = 0.0000000000000001

def plotPoints(plotOutFunction, xyPoints, fy=[]):
	x1 = []
	x2 = []
	y = []
	
	for i in range(len(xyPoints)):
		x1.append(xyPoints[i][0])
		x2.append(xyPoints[i][1])
		y.append(xyPoints[i][len(xyPoints[0])-1])
	
	plt.title("XY points and predicted function")

	isFirst1 = True
	isFirst0 = True
	for i in range(len(x1)):
		if y[i] == 1:
			if isFirst1:
				plt.plot(x1[i],x2[i],'b+',label='Actual output value 1',markersize=12)
				isFirst1 = False
			else:
				plt.plot(x1[i],x2[i],'b+',markersize=12)

		else:
			if isFirst0:
				plt.plot(x1[i],x2[i],'bo',label='Actual output value 0',markersize=12)
				isFirst0 = False
			else:
				plt.plot(x1[i],x2[i],'bo',markersize=12)

	if (plotOutFunction == 1):
		isFirst0 = True
		isFirst1 = True
		#for i in range(len(x1)):
		#	if fy[i] >= 0.5:
		#		if isFirst1:
		#			plt.plot(x1[i],x2[i],'r>',label='Predicted output value 1')
		#			isFirst1 = False
		#		else:
		#			plt.plot(x1[i],x2[i],'r>')
		#	else:
		#		if isFirst0:
		#			plt.plot(x1[i],x2[i],'rs',label='Predicted output value 0')
		#			isFirst0 = False
		#		else:
		#			plt.plot(x1[i],x2[i],'rs')

	bndry = []
	for i in range(len(x1)):
		bndry.append((-1 * thetaVec[2] - thetaVec[0] * x1[i]) / thetaVec[1])
	plt.plot(x1, bndry, '-')

	plt.xlabel("Input Data X1")
	plt.ylabel("Input Data X2")
	plt.legend()

	plt.show()	


def readPoints(fileX,fileY):
	noOfPoints = 0
	xyPoints = []

	f1 = open(fileX, "r")
	f2 = open(fileY, "r")

	xPoint = f1.read().splitlines()
	yPoint = f2.read().splitlines()

	for i in range(len(xPoint)):
		xy1Point = []

		xPoints = xPoint[i].split(",")
		xy1Point.append(float(xPoints[0]))
		xy1Point.append(float(xPoints[1]))
		xy1Point.append(1.0)			# This is the value corresponding to theta0 (since theta0 is the intercept)
		xy1Point.append(float(yPoint[i]))

		xyPoints.append(xy1Point)
		noOfPoints = noOfPoints + 1

	f1.close()
	f2.close()

	return xyPoints, noOfPoints

def normalize(xyPoints, mean, var):
	for i in range(len(xyPoints[0])):
		if (i == len(xyPoints[0])-2):		# Ignore the second last col and last col. Second last col is 1.0 that denotes intercept
			break

		x = []
		for j in range(len(xyPoints)):
			x.append(xyPoints[j][i])

		mean[i]=(np.mean(x))
		var[i]=(np.std(x))

		for j in range(len(xyPoints)):
			xyPoints[j][i] = (xyPoints[j][i] - mean[i]) / var[i]

	return xyPoints, mean, var

def getHTheta(thetaVec, xyPoints, row):
	htheta = 0.0
	for i in range(len(thetaVec)):
		htheta = htheta + thetaVec[i] * xyPoints[row][i]
	
	htheta = 1 / (1 + math.exp(-1 * htheta))
	return htheta

def getPredictedFn(xyPoints, thetaVec):
	fy = []
	y = [0] * len(xyPoints)

	for k in range(len(xyPoints)):
		fy1 = getHTheta(thetaVec, xyPoints, k)
		fy.append(fy1)
		y[k] = xyPoints[k][len(xyPoints[0])-1]

	return fy

def unormalize(xyPoints, mean, var):
	for i in range(len(xyPoints[0])):
		if (i == len(xyPoints[0])-2):
			break

		for j in range(len(xyPoints)):
			xyPoints[j][i] = (xyPoints[j][i] * var[i]) + mean[i]

	return xyPoints

def calculateError(xyPoints, thetaVec):
	sum = 0.0
	for i in range(len(xyPoints)):
		htheta = getHTheta(thetaVec, xyPoints, i)
		yi = xyPoints[i][len(xyPoints[0])-1]
			
		term2 = math.log(1 - htheta)
		term1 = math.log(htheta)
		sum = sum + yi * term1 + (1 - yi) * term2

	Jo = sum / float(len(xyPoints))
	return Jo

def calcDerivateOfError(thetaVec, xyPoints):
	derivJoMat = np.zeros((len(thetaVec), 1))

	for k in range(len(thetaVec)):
		for i in range(len(xyPoints)):
			htheta = getHTheta(thetaVec, xyPoints, i)			
			yi = xyPoints[i][len(xyPoints[0])-1]

			derivJoMat[k][0] = derivJoMat[k][0] + (yi - htheta) * xyPoints[i][k]

		derivJoMat[k][0] = derivJoMat[k][0] / float(len(xyPoints))

	return derivJoMat

def calcInvHessianError(thetaVec, xyPoints):
	hessianJoMat = np.empty((len(thetaVec), len(thetaVec),))
	hessianJoMat[:] = np.nan

	htheta = [0] * (len(xyPoints))

	for l in range(len(thetaVec)):
		for k in range(len(thetaVec)):
			if math.isnan(hessianJoMat[l][k]) == False:
				continue

			hessianJoMat[l][k] = 0
			hessianJoMat[k][l] = 0

			for i in range(len(xyPoints)):
				hthetaVar = 0.0
				if htheta[i] == 0.0:
					hthetaVar = getHTheta(thetaVec, xyPoints, i)
					htheta[i] = hthetaVar
				else:
					hthetaVar = htheta[i]
			
				hessianJoMat[l][k] = hessianJoMat[l][k] + hthetaVar * (1 - hthetaVar) * xyPoints[i][k] * xyPoints[i][l]

			hessianJoMat[l][k] = hessianJoMat[l][k] / float(len(xyPoints))
			hessianJoMat[k][l] = hessianJoMat[l][k]

	hessianJoMat = np.linalg.inv(hessianJoMat)

	return hessianJoMat

def updateTheta(xyPoints, thetaVec):
	thetaMat = np.zeros((len(thetaVec), 1))
	for i in range(len(thetaVec)):
		thetaMat[i][0] = thetaVec[i]
	derivErrorMat = calcDerivateOfError(thetaVec, xyPoints)
	invHessianErrorMat = calcInvHessianError(thetaVec, xyPoints)

	thetaMat = np.add(thetaMat, np.matmul(invHessianErrorMat, derivErrorMat))

	# Once all newTheta is calculated, update the thetaVec in this loop
	for i in range(len(thetaVec)):
		thetaVec[i] = thetaMat[i][0]

	return thetaVec

fileX = sys.argv[1] # prints python_script.py
fileY = sys.argv[2] # prints var1

# LOCAL VARIABLES
xyPoints,noOfPoints = readPoints(fileX,fileY)
thetaVec = [0] * (len(xyPoints[0]) - 1)		# Last col is y so we will subtract that col in thetaVec and last theta is O0 which we will add a column in theta Vec
JPvs = 0
noOfIterations = 0

# GLOBAL VARIABLE
mean = [0] * len(xyPoints[0])
var = [0] * len(xyPoints[0])

xyPoints, mean, var = normalize(xyPoints, mean, var)

while 1:
	thetaVec = updateTheta(xyPoints, thetaVec)
	JCur = calculateError(xyPoints, thetaVec)

	fy = getPredictedFn(xyPoints, thetaVec)
	#plotPoints(1,xyPoints, fy)

	if abs(JCur - JPvs) < threshold:
		break					# Converged

	JPvs = JCur
	noOfIterations = noOfIterations + 1

fy = getPredictedFn(xyPoints, thetaVec)
plotPoints(1,xyPoints,fy)
xyPoints = unormalize(xyPoints, mean, var)

print ("Theta Vector = ", thetaVec)
print ("Number of Iterations = ", noOfIterations)

