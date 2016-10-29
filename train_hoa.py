import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataframe = pandas.read_csv("/tmp/codegit/AWS2/hoa.csv", header=None)
dataset = dataframe.values
filename ="iris data"
print('Loaded data file {0} with {1} rows and {2} columns').format(filename, len(dataset), len(dataset[0]))
print(dataset[0])
print(dataset[1])
X = dataset[:,0:4]
Y = dataset[:,4]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
print encoded_Y
# split into input (X) and output (Y) variables
#X = dataset[:,0:60].astype(float)
#Y = dataset[:,60]
