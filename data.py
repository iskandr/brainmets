

FILENAME = 'BrainMets.xlsx'
MONTHS_TO_LIVE = 9
N_TRAIN = 250

def make_dataset(df, months_to_live = MONTHS_TO_LIVE):
	mask = (df['Dead'] | (df['SurvivalMonths'] >= months_to_live))
	y = np.array((df['Dead'] & (df['SurvivalMonths'] < months_to_live)).ix[mask] == 1)
	Xa = np.array(X.ix[mask]).astype('int')
	return Xa, y

def load_dataframe(filename = FILENAME):
	return pd.read_excel(filename, 'DATA', header=1)
	
def load_dataset(filename = FILENAME, months_to_live = MONTHS_TO_LIVE):
	df = load_dataframe(filename)
	return make_dataset(df, months_to_live)

def split_dataset(df, months_to_live = MONTHS_TO_LIVE, n_train = N_TRAIN, binarize_categorical = True, shuffle = True, verbose = True):
	X, y = make_dataset_arrays(df, months_to_live = months_to_live)
	n_features = Xa.shape[1]
	if binarize_categorical:
		binarize_mask = np.ones(n_features, dtype=bool)
		binarize_mask[0] = False
		binarizer = sklearn.preprocessing.OneHotEncoder(categorical_features = binarize_mask)
		X = binarizer.fit_transform(X).todense()
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

	
