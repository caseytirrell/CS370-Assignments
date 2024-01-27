import numpy as np
import matplotlib.pyplot as plot
from scipy.stats import multivariate_normal

global size 
size = 200

#Question 1 part 1
def normalMultivariate(m, covariance, size):
    mean = np.array(m)
    length = len(mean)
    cholesky = np.linalg.cholesky(covariance)
    standardNormal = np.random.standard_normal((size, length))
    dot = np.dot(standardNormal, cholesky.T) + mean
    return dot

def genBivariatePoints(m, covariance):
    points = normalMultivariate(m, covariance, size)
    return points

def plotPoints(points):
    plot.figure(figsize=(8,8))
    plot.scatter(points[:, 0], points[:, 1], c = 'red')
    plot.title('Problem 1a Plot')
    plot.xlabel('First Variable')
    plot.ylabel('Second Variable')
    plot.show()

#Question 1b
def plotContour(points, mean, covariance):
    x = np.linspace(min(genPoints[:, 0]) - 1, max(genPoints[:, 0]) + 1, 500)
    y = np.linspace(min(genPoints[:, 1]) - 1, max(genPoints[:, 1]) + 1, 500)
    X, Y = np.meshgrid(x, y)
    position = np.dstack((X, Y))
    randomVar = multivariate_normal(mean, covariance)
    
    plot.figure(figsize=(8,8))
    plot.contour(X, Y, randomVar.pdf(position))
    plot.scatter(points[:, 0], points[:, 1], c = 'red')
    plot.title('Problem 1b Plot')
    plot.xlabel('First Variable')
    plot.ylabel('Second Variable')
    plot.legend()
    plot.show()

covarianceMatrix = [[1, 0.5], [0.5, 1]]
meanVector = [0, 0]

genPoints = genBivariatePoints(meanVector, covarianceMatrix)
plotPoints(genPoints) 
plotContour(genPoints, meanVector, covarianceMatrix)