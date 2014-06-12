import pandas as pd
import numpy as np
import sklearn.preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression

FILENAME = 'BrainMets.xlsx'
MONTHS_TO_LIVE = 9
N_TRAIN = 250

def categorical_indices(values):
	"""
	When we have a categorical feature like 'cancer type', we want to transform its unique values
	to indices in some range [0, ..., n-1] where n is the number of categories
	"""
	unique = values.unique()
	indices  = np.zeros(len(values), dtype=int)
	for (i, v) in enumerate(sorted(unique)):
		indices[np.array(values == v)] = i
	return indices 

def load_dataframe(filename = FILENAME):
	df = pd.read_excel(filename, 'DATA', header=1)
	df['cancer type']  = df['cancer type'].str.lower().str.strip()
	# df['cancer type'] = categorical_indices(cancer_type)
	df['Brain Tumor Sx']  = df['Brain Tumor Sx'].astype('float')
	# df['Brain Tumor Sx'] = categorical_indices(brain_tumor_sx)
	return df 

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


def feature_selection(df, Y, training_set_mask):
	Y_training = Y[training_set_mask]
	df_training = df.ix[training_set_mask]
	fields = []
	n_tumors = df['# of tumors']
	n_tumors_training = n_tumors[training_set_mask]

	
def impute(X, df, name, model, postprocess = lambda x: x, maxval = None):
	Y = df[name]
	missing = np.array(Y.isnull())
	X_train = X[~(missing)]
	Y_train = Y[~missing]
	X_test = X[missing]
	model.fit(X_train, Y_train)
	Y_test = model.predict(X_test)
	Y_test = postprocess(Y_test)
	if maxval:
		Y_test = np.minimum(Y_test, maxval)
	Y_filled = Y.copy()
	Y_filled[missing] = Y_test
	df[name] = Y_filled 

def impute_missing_features(df):
	input_fields = df[[
		'Brain Tumor Sx', 
		'RPA', 
		'ECOG', 
		'Prior WBRT', 
		'Diagnosis of Primary at the same time as Brain tumor'
	]]
	X = np.array(input_fields)	
	missing = df['Extracranial Disease Status'].isnull()
	impute(X, df, 'Extracranial Disease Status', LogisticRegression())
	impute(X, df, 'K Score', LinearRegression(), lambda x: 10*(x.astype('int')/10), maxval = 100)

	return df

def extract_features(df, binarize_categorical):
	
	df = df.copy()
	df['log_age']= np.log2(df['age'])
	
	df = impute_missing_features(df)
	
	df['# of tumors > 1'] = df['# of tumors'] > 1
	df['# of tumors > 4'] = df['# of tumors'] > 4
	df['# of tumors > 10'] = df['# of tumors'] > 10

	df['age <45'] =  df['age'] < 45
	df['age 45-55'] = (df['age'] >= 45) & (df['age'] < 55)
	df['age 55-65'] = (df['age'] >= 55) & (df['age'] < 65)
	df['age 65-75'] = (df['age'] >= 65) & (df['age'] < 75)
	df['age >=75'] = (df['age'] >= 75) 
	
	df['age <40'] =  df['age'] < 40
	df['age 40-50'] = (df['age'] >= 40) & (df['age'] < 50)
	df['age 50-60'] = (df['age'] >= 50) & (df['age'] < 60)
	df['age 50-70'] = (df['age'] >= 50) & (df['age'] < 70)
	df['age 60-70'] = (df['age'] >= 60) & (df['age'] < 70)
	df['age 70-80'] = (df['age'] >= 70) & (df['age'] < 80)
	df['age >=80'] = (df['age'] >= 80) 
	df['age >=70'] =df['age'] >= 70
	df['age 45-60'] = (df['age'] >= 45) & (df['age'] < 60)
	df['Normalized K Score'] = df['K Score'] / 100.0
	continuous_fields = [
		'# of tumors > 1',
		'age 50-70', 
		'age >=70',
		'Normalized K Score',
	]
	binary_fields = [
		'Prior WBRT', 
		'Diagnosis of Primary at the same time as Brain tumor'
	]
	9, 12, 14, 15, 16, 18, 20, 22, 25
	categorical_fields = [
		'Extracranial Disease Status',
		'cancer type', 
	    'Brain Tumor Sx', 
		'RPA', 
		'ECOG', 
	]
	vectors = []

	for field in continuous_fields + binary_fields: 
		v = np.array(df[field]).astype('float')
		vectors.append(v)

	for field in categorical_fields:
		values = df[field]
		if binarize_categorical:
			unique = np.unique(values)
			print "Expanding %s into %d indicator variables: %s" % (field, len(unique), unique)
			for i, v in enumerate(sorted(unique)):
				print len(vectors), field, v, np.sum(values == v)
				vec = np.zeros(len(values), dtype='float')
				vec[np.array(values == v)] = 1
				vectors.append(vec)
		else:
			vectors.append(categorical_indices(values))

	X = np.vstack(vectors).T
	print X.dtype, X.shape
	return X

def make_dataset(df, binarize_categorical):
	"""
	Load dataset with continuous outputs
	"""
	
	dead = np.array(df['Dead'] == 1)
	Y = np.array(np.array(df['SurvivalMonths']))

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

# TODO: fill in missing cancer types 
def annotate_5year_survival(df):
	five_year_survival = {
		'breast': 25,
		'nsclc': 4,
		'sclc' : None,
		'rcc' : 12.1,
		'melanoma' : 16.1,
		'carcinoid' : None, 
		'endometrial' : 17.5,
		'sarcoma' : None,
		'colon' : 12.9, 
		'rectal' : None, 
		'prostate' : 28,
		'uterine' : None ,
		'nasopharyngeal' : None, 
		'thyroid' : 54.7, 
	}



	
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


