from readData import readData
from mLPerceptron import mLPerceptron

trainData , trainLabel , testData , testLabel = readData ()

absoluteError , squaredError = mLPerceptron (trainData , trainLabel , testData , testLabel)

print (" erro absoluto médio : " , absoluteError , "\n erro quadrático médio : " , squaredError)