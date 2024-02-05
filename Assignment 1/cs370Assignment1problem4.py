import numpy as np
import matplotlib.pyplot as plot

#Function to create training data for the SGD Algorithm from the notes
def createToyData(func, sampleSize, std, domain = [0, 1]):
    x = np.linspace(domain[0], domain[1], sampleSize)
    np.random.shuffle(x)
    y = func(x) + np.random.normal(scale = std, size = x.shape)
    return x, y

#Sinusodial function from the notes
def sinusodial(x):
    return np.sin(2 * np.pi * x)

#SGD algorithm with the Adam enchancement
class SGDAdam:
    def __init__(self, lr, epochs, bs, tolerance, b1, b2, epsilon):
        self.lr = lr
        self.epochs = epochs
        self.bs = bs
        self.tolerance = tolerance
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon
        self.weights = None
        self.m = None
        self.v = None
        self.t = 0
    
    def training(self, xTrain, yTrain, xValidation, yValidation):
        avgLosses = []
        validationLoss = []
        
        #n is number of samples d is number of traits
        n, d = xTrain.shape 
        self.weights = np.zeros(d)
        self.m = np.zeros(d)
        self.v = np.zeros(d)
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
                self.t += 1
                self.m = self.b1 * self.m + (1 - self.b1) * gradient
                self.v = self.b2 * self.v + (1 - self.b2) * (gradient ** 2)
                m_hat = self.m / (1 - self.b1 ** self.t)
                v_hat = self.v / (1 - self.b2 ** self.t)
                self.weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                
                loss = np.mean(trainingErrors ** 2)
                losses.append(loss)

            prediction = xValidation.dot(self.weights)
            vError = prediction - yValidation.flatten()
            vLoss = np.mean(vError ** 2)
            
            avgLosses.append(np.mean(losses))
            validationLoss.append(vLoss)
       
        return avgLosses, validationLoss

#SGD algorithm with momentum as an enhancement
class SGDMomentum:
    def __init__(self, lr, epochs, bs, tolerance, momentum):
        self.lr = lr
        self.epochs = epochs
        self.bs = bs
        self.tolerance = tolerance
        self.momentum = momentum
        self.weights = None
        self.velocity = None
    
    def training(self, xTrain, yTrain, xValidation, yValidation):
        avgLosses = []
        validationLoss = []
        
        #n is number of samples d is number of traits
        n, d = xTrain.shape 
        self.weights = np.zeros(d)
        self.velocity = np.zeros(d)
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
                
                self.velocity = self.momentum * self.velocity + gradient
                self.weights -= self.lr * self.velocity

                loss = np.mean(trainingErrors ** 2)
                losses.append(loss) 

            prediction = xValidation.dot(self.weights)
            vError = prediction - yValidation.flatten()
            vLoss = np.mean(vError ** 2)
            
            avgLosses.append(np.mean(losses))
            validationLoss.append(vLoss)
            
        return avgLosses, validationLoss
    
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
        
        #n is number of samples d is number of traits
        n, d = xTrain.shape 
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

            prediction = xValidation.dot(self.weights)
            vError = prediction - yValidation.flatten()
            vLoss = np.mean(vError ** 2)
            
            avgLosses.append(np.mean(losses))
            validationLoss.append(vLoss)
            
        return self.weights, avgLosses, validationLoss
    
def movingAvg(data, windowSize):
    movingAvg = []
    for i in range(len(data) - (windowSize + 1)):
        window = data[i:i + windowSize]
        windowAvg = np.mean(window)
        movingAvg.append(windowAvg)
    return movingAvg

#Hyperparamters used
lr = 0.001
epochs = 1000
bs = 32
tolerance = 1e-4
#For momentum
momentum = 0.9
#For Adam
b1 = 0.9
b2 = 0.999
epsilon = 1e-8

xTrain, yTrain = createToyData(sinusodial, 80, 0.25) #80% training data
xValidation, yValidation = createToyData(sinusodial, 20, 0.25) #20% validation

xTrain = np.hstack((np.ones((xTrain.shape[0], 1)), xTrain[:, None]))
yTrain = yTrain[:, None]
xValidation = np.hstack((np.ones((xValidation.shape[0], 1)), xValidation[:, None]))
yValidation = yValidation[:, None]

sgd_adam = SGDAdam(lr, epochs, bs, tolerance, b1, b2, epsilon)
trainingLossAdam, validationLossAdam = sgd_adam.training(xTrain, yTrain, xValidation, yValidation)

sgdMomentum = SGDMomentum(lr, epochs, bs, tolerance, momentum)
trainingLossMomentum, validationLossMomentum = sgdMomentum.training(xTrain, yTrain, xValidation, yValidation)

sgd = SGD(lr, epochs, bs, tolerance)
weights, trainingLoss, validationLoss = sgd.training(xTrain, yTrain, xValidation, yValidation)

windowSize = 100
smoothedTrainingLoss = movingAvg(trainingLoss, windowSize)
smoothedValidationLoss = movingAvg(validationLoss, windowSize)
smoothedTrainingLossMomentum = movingAvg(trainingLossMomentum, windowSize)
smoothedValidationLossMomentum = movingAvg(validationLossMomentum, windowSize)
smoothedTrainingLossAdam = movingAvg(trainingLossAdam, windowSize)
smoothedValidationLossAdam = movingAvg(validationLossAdam, windowSize)

plot.figure(figsize=(8,8))
plot.plot(smoothedTrainingLoss, color="blue", label="Smoothed Training Loss (Baseline SGD)")
plot.plot(smoothedValidationLoss, color="red", label="Smoothed Validation Loss (Baseline SGD)")
plot.plot(smoothedTrainingLossMomentum, color="green", label="Smoothed Training Loss (Momentum)")
plot.plot(smoothedValidationLossMomentum, color="yellow", label="Smoothed Validation Loss (Momentum)")
plot.plot(smoothedTrainingLossAdam, color="purple", label="Smoothed Training Loss (Adam)")
plot.plot(smoothedValidationLossAdam, color="pink", label="Smoothed Validation Loss (Adam)")
plot.title('Loss vs Epoch(Baseline SGD vs Momentum vs Adam)')
plot.xlabel('Epoch')
plot.ylabel('Loss')
plot.legend()
plot.show()
