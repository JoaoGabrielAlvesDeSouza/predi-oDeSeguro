from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

def mLPerceptron (trainData , trainLabel , testData , testLabel) :
  
  perceptron = MLPRegressor (hidden_layer_sizes = (100, 50, 20,) , max_iter = 1000000 , activation = "relu").fit (trainData , trainLabel)

  predictions = perceptron.predict (testData)

  absoluteError = mean_absolute_error (testLabel , predictions)
  squaredError = mean_squared_error (testLabel , predictions)
  
  return absoluteError , squaredError