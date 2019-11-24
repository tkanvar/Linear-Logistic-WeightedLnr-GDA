import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.lines as mLines
import sys

def getQuadraticBoundary(xPoints, yPoints, phi, mu0, mu1, sigma0, sigma1):
	sigma0Inv = np.linalg.inv(sigma0)
	sigma1Inv = np.linalg.inv(sigma1)
	sigma0Det = np.linalg.det(sigma0)
	sigma1Det = np.linalg.det(sigma1)

	A = sigma0Inv - sigma1Inv
	B = -2 * (mu0.T @ sigma0Inv - mu1.T @sigma1Inv)
	C = mu0.T @ sigma0Inv @ mu0 - mu1.T @ sigma1Inv @ mu1 - 2 * np.log (((1 / phi) - 1) * (sigma1Det / sigma0Det))

	return A, B, C

def getLinearBoundary(xPoints, yPoints, mu0, mu1, sigma):
	sigmaInv = np.linalg.inv(sigma)
	diff = mu0 - mu1
	w = sigmaInv @ diff

	term1 = - (mu1.T @ sigmaInv @ mu1) / 2
	term2 = (mu0.T @ sigmaInv @ mu0) / 2
	w0= term1 + term2

	return w, w0

def plotPoints(plotOutFunction, plotQuadFunction, xPoints, yPoints, fw=[], fw0=[], A=[], B=[], C=[], quadpartplot=0):
	x1 = []
	x2 = []
	y = []
	
	for i in range(len(xPoints)):
		x1.append(xPoints[i][0])
		x2.append(xPoints[i][1])
		y.append(yPoints[i])
	
	plt.title("XY points and predicted function")

	isFirst1 = True
	isFirst0 = True
	for i in range(len(x1)):
		if y[i] == 1:
			if isFirst1:
				plt.plot(x1[i],x2[i],'b+',label='Alaska',markersize=12)
				isFirst1 = False
			else:
				plt.plot(x1[i],x2[i],'b+',markersize=12)

		else:
			if isFirst0:
				plt.plot(x1[i],x2[i],'bo',label='Canada',markersize=12)
				isFirst0 = False
			else:
				plt.plot(x1[i],x2[i],'bo',markersize=12)

	if plotOutFunction:
		w0 = fw0[0][0]
		w1 = fw[0][0]
		w2 = fw[1][0]
		bndry = []
		for i in range(len(x1)):
			bndry.append((-w0 - w1*x1[i]) / w2)
		plt.plot(x1, bndry, '-',label='Linear Boundary')

	if plotQuadFunction:
		dummyX0, dummyX1 = np.mgrid[-2.5:2.5:100j, -3:3:100j]			# Part (e)
		if quadpartplot == 1:
			dummyX0, dummyX1 = np.mgrid[-5.5:5.5:100j, -6.5:6.5:100j]				# Part (f)

		dummyX0_1, dummyX1_1 = np.meshgrid(x1, x2)
		points = np.c_[dummyX0.flatten(), dummyX1.flatten()]
		points_1 = np.c_[dummyX0_1.flatten(), dummyX1_1.flatten()]

		def bndry(x):
        		return x.T @ A @ x + B @ x + C

		quad_1 = np.array([bndry(m) for m in points_1]).reshape(dummyX0_1.shape)
		quad = np.array([bndry(m) for m in points]).reshape(dummyX0.shape)
		plt.contour(dummyX0_1, dummyX1_1, quad_1, [0], colors="g")
		plt.contour(dummyX0, dummyX1, quad, [0], colors="y")

	plt.xlabel("Input Data X1")
	plt.ylabel("Input Data X2")
	plt.legend()

	plt.show()

def readPoints(fileX,fileY):
	noOfPoints = 0
	xPoints = []
	yPoints = []

	f1 = open(fileX, "r")
	f2 = open(fileY, "r")

	xPoint = f1.read().splitlines()
	yPoint = f2.read().splitlines()

	for i in range(len(xPoint)):
		x1Point = []
		xPointsInterm = xPoint[i].split("  ")
		x1Point.append(float(xPointsInterm[0]))
		x1Point.append(float(xPointsInterm[1]))

		y1Point = 0			# Alaska - 1, Canada - 0
		if yPoint[i] == 'Alaska':
			y1Point = 1

		xPoints.append(x1Point)
		yPoints.append(y1Point)
		noOfPoints = noOfPoints + 1

	f1.close()
	f2.close()

	return xPoints, yPoints, noOfPoints

def normalize(xPoints):
	mean = [0] * len(xPoints[0])
	var = [0] * len(xPoints[0])

	for i in range(len(xPoints[0])):
		x = []
		for j in range(len(xPoints)):
			x.append(xPoints[j][i])

		mean[i] = (np.mean(x))
		var[i] = (np.std(x))

		for j in range(len(xPoints)):
			xPoints[j][i] = (xPoints[j][i] - mean[i]) / var[i]

	return xPoints

def calculatePhi(yPoints):
	return sum(yPoints) / len(yPoints)

def calculateMean(xPoints, yPoints, classNo):
	xsum = [0] * len(xPoints[0])
	ysum = 0.0
	for i in range(len(yPoints)):
		if yPoints[i] == classNo:
			ysum = ysum + 1
			for j in range(len(xPoints[0])):
				xsum[j] = xsum[j] + xPoints[i][j]

	mu = np.zeros((len(xPoints[0]), 1))
	for i in range(len(xPoints[0])):
		mu[i][0] = (xsum[i] / ysum)

	return mu

def calculateSigma(xPoints, mu, yPoints, classNo):
	sum = np.zeros((len(xPoints[0]), len(xPoints[0])))
	temp = np.zeros((len(xPoints[0]), 1))
	tempT = np.zeros((1, len(xPoints[0])))
	noOfY = 0

	for i in range(len(xPoints)):
		if yPoints[i] == classNo:
			for j in range(len(xPoints[0])):
				temp[j][0] = xPoints[i][j] - mu[j]
				tempT[0][j] = temp[j][0]

			sum[0][0] = sum[0][0] + temp[0][0] * tempT[0][0]
			sum[0][1] = sum[0][1] + temp[0][0] * tempT[0][1]
			sum[1][0] = sum[1][0] + temp[1][0] * tempT[0][0]
			sum[1][1] = sum[1][1] + temp[1][0] * tempT[0][1]

			noOfY = noOfY + 1

	for i in range(len(xPoints[0])):
		for j in range(len(xPoints[0])):
			sum[i][j] = sum[i][j] / noOfY

	return sum

def calculateCommonSigma(xPoints, mu0, mu1, yPoints):
	sum = np.zeros((len(xPoints[0]), len(xPoints[0])))
	temp = np.zeros((len(xPoints[0]), 1))
	tempT = np.zeros((1, len(xPoints[0])))

	for i in range(len(xPoints)):
		if yPoints[i] == 1:
			for j in range(len(xPoints[0])):
				temp[j][0] = xPoints[i][j] - mu1[j]
				tempT[0][j] = temp[j][0]
		else:
			for j in range(len(xPoints[0])):
				temp[j][0] = xPoints[i][j] - mu0[j]
				tempT[0][j] = temp[j][0]

		sum[0][0] = sum[0][0] + temp[0][0] * tempT[0][0]
		sum[0][1] = sum[0][1] + temp[0][0] * tempT[0][1]
		sum[1][0] = sum[1][0] + temp[1][0] * tempT[0][0]
		sum[1][1] = sum[1][1] + temp[1][0] * tempT[0][1]

	for i in range(len(xPoints[0])):
		for j in range(len(xPoints[0])):
			sum[i][j] = sum[i][j] / len(xPoints)

	return sum

fileX = sys.argv[1] # prints python_script.py
fileY = sys.argv[2] # prints var1
partToExe = sys.argv[3]

xPoints,yPoints,noOfPoints = readPoints(fileX,fileY)
xPoints = normalize(xPoints)

if partToExe == "0":

	############## Part (a)
	phi = calculatePhi(yPoints)
	mu0 = calculateMean(xPoints, yPoints, 0)
	mu1 = calculateMean(xPoints, yPoints, 1)
	sigma = calculateCommonSigma(xPoints, mu0, mu1, yPoints)

	print ("Phi = ", phi)
	print ("Mu0 = ", mu0)
	print ("Mu1 = ", mu1)
	print ("Sigma = ", sigma)

	############# Part (b)
	plotPoints(False, False, xPoints, yPoints)

	############# Part (c)
	w, w0 = getLinearBoundary(xPoints, yPoints, mu0, mu1, sigma)
	plotPoints(True, False, xPoints, yPoints, w, w0)

else:

	############# Part (d)
	phi = calculatePhi(yPoints)
	mu0 = calculateMean(xPoints, yPoints, 0)
	mu1 = calculateMean(xPoints, yPoints, 1)
	sigma0 = calculateSigma(xPoints, mu0, yPoints, 0)
	sigma1 = calculateSigma(xPoints, mu0, yPoints, 1)

	print ("Phi = ", phi)
	print ("Mu0 = ", mu0)
	print ("Mu1 = ", mu1)
	print ("Sigma0 = ", sigma0)
	print ("Sigma1 = ", sigma1)

	############ Part (e) and (f)

	A, B, C = getQuadraticBoundary(xPoints, yPoints, phi, mu0, mu1, sigma0, sigma1)
	plotPoints(False, True, xPoints, yPoints, [], [], A, B, C, 0)
	plotPoints(False, True, xPoints, yPoints, [], [], A, B, C, 1)
