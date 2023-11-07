import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

def readData () :

  data = p.read_csv ("insurance.csv")

  data = p.get_dummies (data , columns = ["sex" , "smoker" , "region"])

  trainData , testData = train_test_split (data , test_size = 0.4 , random_state = 3000 , shuffle = True)
  
  trainLabel = trainData ["charges"]
  del trainData ["charges"]

  testLabel = testData ["charges"]
  del testData ["charges"]

  trainData = normalize (trainData , axis = 0)
  testData = normalize (testData , axis = 0)

  return trainData , trainLabel , testData , testLabel