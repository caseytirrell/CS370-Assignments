import numpy as np
from scipy.linalg import svd

global sampleSize
sampleSize = 1000

#function that uses the svd
def principalComponents(covarianceMatrix):
    U, _, _ = svd(covarianceMatrix)
    principalComps = U[:, :2]
    return principalComps

def guassainVectors(mean, covarianceMatrix):
    guassian = np.random.multivariate_normal(mean, covarianceMatrix, sampleSize)
    return guassian

def createProjection(vectors, principalComps):
    mapping = vectors @ principalComps
    transpose = mapping @ principalComps.T
    return transpose

def generateVectors(covarianceMatrix):
    default = np.zeros(covarianceMatrix.shape[0])
    gVectors = guassainVectors(default, covarianceMatrix)
    return gVectors

def projectVectors(guassianVectors, covarianceMatrix):
    principalComps = principalComponents(covarianceMatrix)
    projectedVectors = createProjection(guassianVectors, principalComps)
    return projectedVectors

covMatrix = np.array([[4, 2, 1], [2, 3, 1.5], [1, 1.5, 2]])

originalVectors = generateVectors(covMatrix)
projectedVectors = projectVectors(originalVectors, covMatrix)

#Average Vectors
avgOriginal = np.mean(originalVectors, axis=0)
avgProjected = np.mean(projectedVectors, axis=0)

for i in range(sampleSize):
    print("Sample Number " + str({i + 1}))
    print("Original Vector " + ": " + str(originalVectors[i]))
    print("Projected Vector " + ": " + str(projectedVectors[i]) + "\n")
print("Sample Size: " + str(sampleSize))
print("Average Original Vector: " + str(avgOriginal))
print("Average Projected Vector: " + str(avgProjected) + "\n")
