import numpy as np
from numpy import linalg as linAlg
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

def plotPoints(plotOutFunction, fy=[], fx=[],tau=-1):
	x = []
	y = []
	
	for i in range(len(xyPoints)):
		x.append(xyPoints[i][0])
		y.append(xyPoints[i][len(xyPoints[0])-1])
	
	if tau == -1:
		plt.title("XY points and predicted function for Linear Regression")
	else:
		plt.title("XY points and predicted function for weighted Regression with tau = " + str(tau))
	plt.scatter(x,y,color='blue',label='Original Plot')
	if (plotOutFunction == 1):
		plt.scatter(fx,fy,color='red',label='Predicted Plot')

	plt.xlabel("Input Data")
	plt.ylabel("Output Data")
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
		xy1Point.append(float(xPoint[i]))
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

	return xyPoints,mean,var

def getPredictedY(xyPoints, thetaVec):
	fy = []
	y = [0] * len(xyPoints)

	for k in range(len(xyPoints)):
		fy1 = 0.0
		for i in range(len(thetaVec)):
			fy1 = fy1 + thetaVec[i] * xyPoints[k][i]

		fy.append(fy1)
		y[k] = xyPoints[k][len(xyPoints[0])-1]

	return fy

def unormalize(xyPoints, mean, var, pfx=[], isFxAvailable=False):
	fx = pfx
	if isFxAvailable == False:
		fx = [0] * len(xyPoints)

	for i in range(len(xyPoints[0])):
		if (i == len(xyPoints[0])-2):
			break

		for j in range(len(xyPoints)):
			xyPoints[j][i] = (xyPoints[j][i] * var[i]) + mean[i]

			if isFxAvailable == True:
				fx[j] = (fx[j] * var[i]) + mean[i]
			else:
				fx[j] = xyPoints[j][i]

	return xyPoints, fx

def calculateTheta(xyPoints, thetaVec, weights, isWeightedLinearReg):
	X = np.zeros((len(xyPoints), 2))
	Xt = np.zeros((2, len(xyPoints)))
	Y = np.zeros((len(xyPoints), 1))
	W = np.zeros((len(xyPoints), len(xyPoints)))

	for i in range(len(xyPoints)):
		for j in range(len(xyPoints[i])-1):
			X[i][j] = xyPoints[i][j]
			Xt[j][i] = xyPoints[i][j]

	for i in range(len(weights)):
		W[i][i] = weights[i]

	for i in range(len(xyPoints)):
		Y[i][0] = xyPoints[i][len(xyPoints[0])-1]

	thetaVec = linAlg.inv(Xt @ X) @ Xt @ Y
	if isWeightedLinearReg == True:
		thetaVec = linAlg.inv(Xt @ W @ X) @ Xt @ W @ Y

	thetaVecRet = [0] * (len(thetaVec))
	for i in range(len(thetaVec)):
		thetaVecRet[i] = thetaVec[i][0]

	return thetaVecRet

def getWeights(xyPoints, noOfPoints, pointToProcess, weights, tau):
	for i in range(noOfPoints):
		diff = 0.0
		for j in range(len(xyPoints[i])-1):
			diff = diff + (pointToProcess[j] - xyPoints[i][j]) * (pointToProcess[j] - xyPoints[i][j])

		weights[i] = math.exp(-1 * (diff / (2 * tau * tau)))

	return weights

def linearRegression(xyPoints, noOfPoints, pointToProcess, isWeightedLinearReg, tau = 0):	
	thetaVec = [0] * (len(xyPoints[0]) - 1)		# Last col is y so we will subtract that col in thetaVec and last theta is O0 which we will add a column in theta Vec
	weights = [1] * (noOfPoints)

	if isWeightedLinearReg == True:
		weights = getWeights(xyPoints, noOfPoints, pointToProcess, weights, tau)
	
	thetaVec = calculateTheta(xyPoints, thetaVec, weights, isWeightedLinearReg)
	fy = getPredictedY(xyPoints, thetaVec)
	
	return fy,thetaVec
	
def weightedLinearRegression(xyPoints, noOfPoints, tau):
	fyFinal = [1] * (noOfPoints)
	xFinal = [1] * (noOfPoints)
	cntr = 0

	minEle = 100000
	maxEle = -100000
	for i in range(len(xyPoints)):
		if minEle > xyPoints[i][0]:
			minEle = xyPoints[i][0]
		if maxEle < xyPoints[i][0]:
			maxEle = xyPoints[i][0]
	points = np.linspace(minEle, maxEle, noOfPoints)
	#for i in range(len(xyPoints)):
	#	points[i] = xyPoints[i][0]

	while cntr < noOfPoints:
		pointToProcess = [1] * (len(xyPoints[0])-1)
		pointToProcess[0] = points[cntr]
		
		fy,thetaVec = linearRegression(xyPoints, noOfPoints, pointToProcess, True, tau)		# True - Is weighted linear regression

		predictedPointToProcess = 0.0
		for j in range(len(thetaVec)):
			predictedPointToProcess = predictedPointToProcess + thetaVec[j] * pointToProcess[j]

		fyFinal[cntr] = predictedPointToProcess
		xFinal[cntr] = pointToProcess[0]

		cntr = cntr + 1

	return fyFinal,xFinal

fileX = sys.argv[1] # prints python_script.py
fileY = sys.argv[2] # prints var1
tau = float(sys.argv[3]) # prints var2

xyPoints,noOfPoints = readPoints(fileX, fileY)

mean = [0] * len(xyPoints[0])
var = [0] * len(xyPoints[0])

######### (a)
xyPoints,mean,var = normalize(xyPoints, mean, var)

fy,thetaVec = linearRegression(xyPoints, noOfPoints, [], False)					# False - Is not weighted linear regression

xyPoints,fx = unormalize(xyPoints, mean, var, [], False)
plotPoints(1,fy,fx)

######### (b)
xyPoints,mean,var = normalize(xyPoints, mean, var)

#tau =0.8
fy,fx = weightedLinearRegression(xyPoints, noOfPoints, tau)

xyPoints,fx = unormalize(xyPoints, mean, var, fx, True)
plotPoints(1,fy, fx,tau)

######### (c)
tauList = [0.1, 0.3, 2, 10]
for i in range(len(tauList)):
	xyPoints,mean,var = normalize(xyPoints, mean, var)

	tau = tauList[i]
	fy,fx = weightedLinearRegression(xyPoints, noOfPoints, tau)

	xyPoints,fx = unormalize(xyPoints, mean, var, fx, True)
	plotPoints(1,fy,fx,tau)
