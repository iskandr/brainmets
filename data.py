import pandas as pd
import numpy as np
import sklearn.preprocessing

FILENAME = 'BrainMets.xlsx'
MONTHS_TO_LIVE = 9
N_TRAIN = 250

def get_expert_predictions(df):
	expert_predictions = {}
	experts = [
		'Prediction(Cleveland Clinic)', 
		' Prediction (Lanie Francis)', 
		'Prediction(Flickinger)', 
		'Prediction(Loefler', 
		'Prediction(Knisely)', 
		'Prediction(Lunsford)', 
		'Prediction (Tahrini)', 
		'Prediction (Sheehan)', 
		'Prediction (Linskey)', 
		'Prediction(friedman)', 
		'Prediction(Stupp)', 
		'Prediction(Rakfal)', 
		'Prediction(Rush)', 
		' Prediction( Kondziolka)'
	]
	for expert in experts:
		expert_predictions[expert] = df[expert]
	return expert_predictions

def extract_features(df, binarize_categorical):
	df = df.copy()
	df['# of tumors > 1'] = df['# of tumors'] > 1
	df['age 30s'] = (df['age'] >= 30) & (df['age'] < 40)
	df['age 40s'] = (df['age'] >= 40) & (df['age'] < 50)
	df['age 50s'] = (df['age'] >= 50) & (df['age'] < 60)
	df['age 60s'] = (df['age'] >= 60) & (df['age'] < 70)
	df['age 70s'] = (df['age'] >= 70) & (df['age'] < 80)
	df['age 80s'] = df['age'] >= 80

	fields = [ 
		'# of tumors > 1', 
		#'age',
		'age 30s',
		'age 40s',
		'age 50s',
		'age 60s',
		'age 70s',
		'age 80s',
		'cancer type', 
		'ECOG', 
		'Prior WBRT', 
		'Brain Tumor Sx', 
		'RPA', 
		'Diagnosis of Primary at the same time as Brain tumor'
	]
	X_raw = df[fields]
	X = np.array(X_raw).astype('int')
	if binarize_categorical:
		n_features = X.shape[1]
		binarize_mask = np.ones(n_features, dtype=bool)
		# age 
		binarize_mask[0:7] = False
		# number of tumors 
		#binarize_mask[7] = False
		
		binarizer = sklearn.preprocessing.OneHotEncoder(categorical_features = binarize_mask)
		X = binarizer.fit_transform(X).todense()
		print np.sum(X[:, -1])
	return X
def make_dataset(df, binarize_categorical):
	"""
	Load dataset with continuous outputs
	"""
	
	dead = df['Dead']	
	Y = np.array(df['SurvivalMonths'])
	expert_predictions = get_expert_predictions(df)
	test_set_mask = np.zeros(len(df), dtype=bool)
	# training set is any data point for which we have no expert
	# predictions 
	for expert_Y in expert_predictions.values():
		test_set_mask |= ~expert_Y.isnull()

	X = extract_features(df, binarize_categorical)
	return X, Y, dead, expert_predictions, test_set_mask

def make_labeled_dataset(df, months_to_live = MONTHS_TO_LIVE, binarize_categorical = True):
	X, Y_continuous, dead, expert_predictions, test_set_mask = make_dataset(df, binarize_categorical)
	# get rid of patients for whom we don't have a long enough history
	mask = np.array(dead | (Y_continuous >= months_to_live))
	X = X[mask]
	Y = dead[mask] & (Y_continuous[mask] < months_to_live)
	return X, Y

def load_dataframe(filename = FILENAME, cancer_types_to_numbers = True):
	df = pd.read_excel(filename, 'DATA', header=1)
	if cancer_types_to_numbers:
		df['cancer type']  = df['cancer type'].str.lower()
		cancer_types = df['cancer type'].unique()
		cancer_type_col = np.zeros(len(df['cancer type']), dtype=int)
		for (i, cancer_type) in enumerate(cancer_types):
			cancer_type_col[np.array(df['cancer type'] == cancer_type)] = i
		df['cancer type'] = cancer_type_col
	return df 

	
def load_dataset(filename = FILENAME, binarize_categorical = True):
	df = load_dataframe(filename)
	return make_dataset(df, binarize_categorical = binarize_categorical)

def load_labeled_dataset(filename = FILENAME, months_to_live = MONTHS_TO_LIVE, binarize_categorical = True):
	df = load_dataframe(filename)
	return make_labeled_dataset(df, months_to_live, binarize_categorical = binarize_categorical)

def split_labeled_dataset(df, months_to_live = MONTHS_TO_LIVE, n_train = N_TRAIN, binarize_categorical = True, shuffle = True, verbose = True):
	X, y = make_labeled_dataset(df, months_to_live = months_to_live, binarize_categorical = binarize_categorical)
	if shuffle:
		idx = np.arange(len(y))
		np.random.shuffle(idx)
		y = y[idx]
		X = X[idx]

	Xtrain = X[:n_train]
	Ytrain = y[:n_train]
	Xtest = X[n_train:]
	Ytest = y[n_train:]
	if verbose:
		print Xtest[[0,1,2], :]
		print Ytest[[0,1,2]]
		print np.mean(Ytrain)
		print np.mean(Ytest)
		print Xtrain.shape
		print Xtest.shape
	return Xtrain, Ytrain, Xtest, Ytest


def load_dataset_splits(filename = FILENAME, months_to_live = MONTHS_TO_LIVE, n_train = N_TRAIN):
	df = load_dataframe(filename)
	return split_dataset(df, months_to_live, n_train)

	
