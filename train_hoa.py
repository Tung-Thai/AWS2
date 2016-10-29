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
df = pd.read_csv("/tmp/codegit/AWS2/cancer_predict.csv",header = None)
dataset=df.values
X=dataset[:,2:33]
Y=dataset[:,1:2]
print dataset[568]
#conver str class colum to interger
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)
print dummy_y
#dummy_y.shape
# define 10-fold cross validation test harness
# define baseline model
# convert class column to int// colum cuoi la class/labale



