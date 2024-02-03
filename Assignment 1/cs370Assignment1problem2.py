import numpy as np
from scipy.linalg import svd
import plotly.graph_objects as plot
from plotly.subplots import make_subplots

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

#Problem 2C
def plotVectors(originalVectors, projectedVectors):
    xMin = np.min(projectedVectors[:, 0])
    xMax = np.max(projectedVectors[:, 0])
    yMin = np.min(projectedVectors[:, 1])
    yMax = np.max(projectedVectors[:, 1])
    zero = 0
    
    figure = plot.Figure()
    figure.add_trace(plot.Scatter3d(x = originalVectors[:, 0], y = originalVectors[:, 1], z = originalVectors[:, 2],
                                    name = 'Original Vectors', mode = 'markers', marker = dict(size = 5)))
    figure.add_trace(plot.Scatter3d(x = projectedVectors[:, 0], y = projectedVectors[:, 1], z = np.zeros(projectedVectors.shape[0]),
                                    name = 'Projected Vectors', mode = 'markers', marker = dict(size = 4)))
    figure.update_layout(
        scene=dict(
            xaxis_title = 'X',
            yaxis_title = 'Y',
            zaxis_title = 'Z'
        ),
        title = 'Original & Projected Vectors'
    )
    figure.add_trace(plot.Mesh3d(
        x = [xMin, xMin, xMax, xMax, xMin, xMin, xMax, xMax],
        y = [yMin, yMax, yMax, yMin, yMin, yMax, yMax, yMin],
        z = [zero, zero, zero, zero, zero, zero, zero, zero],
        name = "Projected Vector Highlight",
        opacity=0.5,
        color = 'coral',
    ))

    figure.show()

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

plotVectors(originalVectors, projectedVectors)

#Problem 2A
#What determines the principal components?
#The principal components are determined by the eigenvectors of the
#covariance matrix that are calculated through the use of Single Value
#Decomposition on the covariance matrix.


#Problem 2B
#What determines the positive or negative correleations between the components?
#The negative and positive correlations between the components is determined by
#by how the variables interact with one another which is encapsulated within
#the covariance matrix.

#Problem 2C
#The plot produced by the algorithm agrees with the positive correlations observed
#by the original covariance matrix. You can tell this by the shape of the projected
#vectors 2D plot.
