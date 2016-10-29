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
# load pima indians dataset
dataframe = pandas.read_csv("/tmp/codegit/AWS2/hoa-header.csv", header=None)
dataset = dataframe.values


# split into input (X) and output (Y) variables
X = dataset[:,0:5]
Y = dataset[:,5]
#conver str class colum to interger
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, encoded_Y):
  # create model
	model = Sequential()
	
	model.add(Dense(12, input_dim=4, init='normal', activation='relu'))
	model.add(Dense(5, init='normal', activation='relu'))
	model.add(Dense(1, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	model.fit(X[train], encoded_Y[train], nb_epoch=20, batch_size=6, verbose=0)
	# evaluate the model
	scores = model.evaluate(X[test], encoded_Y[test], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
