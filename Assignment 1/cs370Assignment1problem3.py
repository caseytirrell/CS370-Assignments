import numpy as np
import matplotlib.pyplot as plot

#Problem 3
#Function to create training data for the SGD Algorithm from the notes
def createToyData(func, sampleSize, std, domain = [0, 1]):
    x = np.linspace(domain[0], domain[1], sampleSize)
    np.random.shuffle(x)
    y = func(x) + np.random.normal(scale = std, size = x.shape)
    return x, y

#Sinusodial function from the notes
def sinusodial(x):
    return np.sin(2 * np.pi * x)

class SGD:
    def __init__(self, lr, epochs, bs, tolerance):
        self.lr = lr
        self.epochs = epochs
        self.bs = bs
        self.tolerance = tolerance
        self.weights = None
    
    def training(self, xTrain, yTrain, xValidation, yValidation):
        avgLosses = []
        validationLoss = []
        
        n, d = xTrain.shape #n is number of samples d is number of traits
        self.weights = np.zeros(d)
        for i in range(self.epochs):
            indices = np.random.permutation(n)
            
            losses = []
            xShuffle = xTrain[indices]
            yShuffle = yTrain[indices].flatten()
            for j in range(0, n, self.bs):
                end = min(j + self.bs, n)
                
                xBatch = xShuffle[j:end]
                yBatch = yShuffle[j:end]

                xLength = len(xBatch)

                hypothesis = xBatch.dot(self.weights)
                trainingErrors = hypothesis - yBatch
                gradient = xBatch.T.dot(trainingErrors) / xLength
                self.weights -= self.lr * gradient

                loss = np.mean(trainingErrors ** 2)
                losses.append(loss)

                grad = np.linalg.norm(gradient)
                
                if grad < self.tolerance:
                    break
            
            avgLosses.append(np.mean(losses))    

            prediction = xValidation.dot(self.weights)
            vError = prediction - yValidation.flatten()
            vLoss = np.mean(vError ** 2)
            validationLoss.append(vLoss)
            
        return self.weights, avgLosses, validationLoss
    
    def predict(self, x):
        return x.dot(self.weights)

def movingAvg(data, windowSize):
    movingAvg = []
    for i in range(len(data) - (windowSize + 1)):
        window = data[i:i + windowSize]
        windowAvg = np.mean(window)
        movingAvg.append(windowAvg)
    return movingAvg

#hyperparameters
lr = 0.01
epochs = 1000
bs = 32
tolerance = 1e-4

#Notes say:
xTrain, yTrain = createToyData(sinusodial, 80, 0.25) #80% training data
xValidation, yValidation = createToyData(sinusodial, 20, 0.25) #20% validation

#adding bias
xTrain = np.hstack((np.ones((xTrain.shape[0], 1)), xTrain[:, None]))
yTrain = yTrain[:, None]
xValidation = np.hstack((np.ones((xValidation.shape[0], 1)), xValidation[:, None]))
yValidation = yValidation[:, None]
    
sgd = SGD(lr, epochs, bs, tolerance)
weights, trainingLoss, validationLoss = sgd.training(xTrain, yTrain, xValidation, yValidation)

#Moving average on training loss to smooth out the line
windowSize = 100
smoothedTrainingLoss = movingAvg(trainingLoss, windowSize)
smoothedValidationLoss = movingAvg(validationLoss, windowSize)
#Plotting the loss vs epoch
plot.figure(figsize=(8,8))
plot.plot(smoothedTrainingLoss, color = "blue", label = "Smoothed Training Loss")
plot.plot(smoothedValidationLoss, color = "red", label = "Smoothed Validation Loss")
plot.title('Loss vs Epoch')
plot.xlabel('Epoch')
plot.ylabel('Loss')
plot.legend()
plot.show()