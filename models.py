import data 
import numpy as np 
import sklearn.linear_model
import sklearn.naive_bayes 
import sklearn.cross_validation
import sklearn.ensemble
import sklearn.decomposition
import sklearn.naive_bayes
import sklearn.feature_selection
                
def load_datasets(months_to_live_start = 1, months_to_live_stop = 15, binarize_categorical = True):
	Xs = {}
	Ys = {}
	for i in xrange(months_to_live_start, months_to_live_stop + 1):
		print "Loading dataset for %d months to live" % i 
		X, Y = data.load_dataset(months_to_live = i, binarize_categorical = binarize_categorical)
		Xs[i] = X
		Ys[i] = Y
	return Xs, Ys
"""
def level_sets_to_continuous(predictors, Xs, Ys)
	keys = list(sorted(Xs.keys()))
	for i in keys:
		left = ...
		right = ... 	
		optimize 1-mean(left) + mean(right)
"""
def binary_predictors(start = 3, stop = 20):
	for i in xrange(start, stop):
		X, Y = data.load_labeled_dataset(months_to_live = i, binarize_categorical = True)
		model = sklearn.linear_model.LogisticRegression()
		aucs = sklearn.cross_validation.cross_val_score(model, X, Y, cv = 10, scoring='roc_auc')
		auc = np.mean(aucs)
		print "Months to live = %d, n_deceased = %d, ROC AUC = %0.4f" % (i, np.sum(Y), auc)


def error(Y_true, Y_pred):
	return np.mean(np.abs(Y_pred - Y_true))

def average_expert_error(experts, Y, deceased = None):
	if deceased is not None:
		Y = Y[deceased]
	Y_expert_combined = np.zeros_like(Y)
	Y_expert_count = np.zeros_like(Y, dtype=int)

	for expert in experts:
		Y_pred = experts[expert]
		if deceased is not None:
			Y_pred = Y_pred[deceased]
		mask = np.array(~(Y_pred.isnull()))
		print expert.strip(), "n =", np.sum(mask)
		Y_pred_subset = np.array(Y_pred[mask].astype('float'))
		print "-- %0.4f" % error(Y[np.array(mask)], Y_pred_subset)
		Y_expert_combined[mask] += Y_pred_subset
		Y_expert_count[mask] += 1

	combined_mask = Y_expert_count > 0
	Y_expert_combined = Y_expert_combined[combined_mask]
	Y_expert_combined /= Y_expert_count[combined_mask]
	return error(Y[combined_mask], Y_expert_combined)

if __name__ == '__main__':

	X, Y, deceased, experts, test_mask = data.load_dataset(binarize_categorical = True)

	

	print "Data shape", X.shape

	print "---"
	print "Average prediction error = %0.4f" % average_expert_error(experts, Y, deceased)
	print "---"

	
	shuffle_data = False
	if shuffle_data:
		random_index = np.arange(len(Y))
		np.random.shuffle(random_index)
		X = np.array(X)
		Y = np.array(Y)
		X = X[random_index]
		Y = Y[random_index]
		deceased = deceased[random_index]
		test_mask = test_mask[random_index]

	train_mask = ~test_mask	

	X_train_full = X[train_mask]
	Y_train_full = Y[train_mask]
	train_deceased = deceased[train_mask]

	X_test_full = X[test_mask]
	Y_test_full = Y[test_mask]
	test_deceased = deceased[test_mask]

	# drop features which are always the same value for 
	# deceased patients in the training set 
	drop_all_same_features = True 
	if drop_all_same_features:
		all_same_mask = np.std(X_train_full[train_deceased], axis=0) == 0
		print "Dropping features", list(np.where(all_same_mask))
		X_train_full = X_train_full[:, ~all_same_mask]
		X_test = X_test_full[:, ~all_same_mask]

	feature_elimination = True 
	if feature_elimination:
		selected = np.zeros(X_train_full.shape[1], dtype=int)
		for i in xrange(1, 10):
			rfe = sklearn.feature_selection.RFECV(estimator = sklearn.linear_model.LogisticRegression())
			rfe.fit(X_train_full[train_deceased], Y_train_full[train_deceased] > i)
			selected += np.array(rfe.support_)
		print "RFE selected features", list(np.where(selected > 0)), "counts", selected
		X_train_full = X_train_full[:, selected > 0]
		X_test_full = X_test_full[:, selected > 0]


	seen_set = set([])
	n_seen = 0
	for i in xrange(len(Y_train_full)):
		v = tuple(X_train_full[i, :])
		if v in seen_set:
			print "Duplicate patient", v
			n_seen += 1
		seen_set.add(v)
	print "# of duplicate training samples: %d" % n_seen

	deceased_only = True 
	if deceased_only:
		X_train = X_train_full[train_deceased]
		Y_train = Y_train_full[train_deceased]
		X_test = X_test_full[test_deceased]
		Y_test = Y_test_full[test_deceased]
	else:
		X_train = X_train_full 
		X_test = X_test_full
		Y_train = Y_train_full
		Y_test = Y_test_full


		
	n_test = len(Y_test_full)
	n_train = len(Y_train) 
	print "Training set = %d, test set = %d, n_features = %d" % (n_train, n_test, X_test.shape[1])
	
	pca_transform = False 
	if pca_transform:
		pca = sklearn.decomposition.PCA(10)
		X_train_full = pca.fit_transform(X_train_full)
		X_test_full = pca.transform(X_test_full)
		X_train = pca.transform(X_train)
		X_test = pca.transform(X_test)
		print "Training size after PCA: %s" % (X_train.shape,)
		
	def fit(model):
		model.fit(X_train, Y_train)
		train_error = error(Y_train,  model.predict(X_train))
		test_error = error(Y_test, model.predict(X_test))
		
		def cv_scorer(model, x, y):
			pred = model.predict(x)
			return error(y, pred)


		cv_scores = sklearn.cross_validation.cross_val_score(model, X_train, Y_train, cv=5, scoring = cv_scorer)
		cv_error = np.mean(cv_scores) 
		print "%s" % (model.__class__.__name__)
		print "-- training error: %0.4f" % train_error
		print "-- CV error: %0.4f" % cv_error
		print "-- test error %0.4f" % test_error

	class AlwaysAverage(object):

		def __init__(self, average = None):
			self.average = average 

		def get_params(self, deep = False):
			return {'average' : self.average}

		def fit(self, X_train, Y_train):
			self.average =  np.mean(Y_train)
		def predict(self, _):
			return self.average 

	fit(AlwaysAverage())	
	fit(sklearn.svm.SVR())
	fit(sklearn.linear_model.RidgeCV(alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]))
	fit(sklearn.linear_model.LassoCV())
	fit(sklearn.linear_model.OrthogonalMatchingPursuitCV())
	fit(sklearn.ensemble.RandomForestRegressor(n_estimators = 200, max_features = 'sqrt'))
	fit(sklearn.ensemble.ExtraTreesRegressor(n_estimators = 200, max_features = 'sqrt'))

	for n in xrange(1, 20):

		train_mask = train_deceased | (Y_train_full >= n)
		x_train = X_train_full[train_mask]
		y_train = Y_train_full[train_mask] >= n
		
		test_mask = test_deceased | (Y_test_full >= n)
		x_test = X_test_full[test_mask]
		y_test = Y_test_full[test_mask] >= n
		
		lr = sklearn.linear_model.LogisticRegression()
		lr.fit(x_train, y_train)
		pred = lr.predict(x_test)

		rf = sklearn.ensemble.RandomForestClassifier(n_estimators = 150)
		rf.fit(x_train, y_train)
		rf_pred = rf.predict(x_test)
		rf_prob = rf.predict_proba(x_test)

		svm1 = sklearn.svm.SVC(probability = True, kernel = 'linear', C = 1)
		svm1.fit(x_train, y_train)
		svm_pred = svm1.predict(x_test)

		

		svm2 = sklearn.svm.SVC(probability = True, kernel = 'linear', C = 10)
		svm2.fit(x_train, y_train)

		svm3 = sklearn.svm.SVC(probability = True, kernel = 'linear', C = 0.1)
		svm3.fit(x_train, y_train)


		probs = svm1.predict_proba(x_test) + svm2.predict_proba(x_test) + svm3.predict_proba(x_test)
		svm_ensemble_pred = np.argmax(probs, axis=1)
		
		ensemble_prob = (probs + rf.predict_proba(x_test) + lr.predict_proba(x_test)) / 5
		ensemble_pred = np.argmax(ensemble_prob, axis=1)
		print n, "n_train", np.sum(y_train), "/", len(y_train), "n_test", np.sum(y_test), "/", len(y_test)
		print "--", "baseline train", np.mean(y_train), "baseline test", np.mean(y_test)
		print "--", "lr", np.mean(pred == y_test), "rf", np.mean(rf_pred == y_test), "svm", np.mean(svm_pred == y_test)
		print "--", "svm ensemble", np.mean(svm_ensemble_pred == y_test), "full ensemble", np.mean(ensemble_pred == y_test)
		