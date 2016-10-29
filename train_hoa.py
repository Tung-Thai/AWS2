#http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
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
# load pima indians dataset
#using header=none do not use header=0 if not you will lost first row

#df = pd.read_csv("/tmp/codegit/AWS2/cancer_predict.csv",header = None)

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

filename = 'iris.csv'
dataset=pd.read_csv("/tmp/codegit/AWS2/cancer_predict.csv",header = None)
print('Loaded data file {0} with {1} rows and {2} columns').format(filename, len(dataset), len(dataset[0]))
print(dataset[0])
# convert string columns to float
for i in range(2,569):
	str_column_to_float(dataset, i)
# convert class column to int
lookup = str_column_to_int(dataset, 1)
print(dataset[0])
print(lookup)
#dummy_y.shape
# define 10-fold cross validation test harness
# define baseline model
# convert class column to int// colum cuoi la class/labale



