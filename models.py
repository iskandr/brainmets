import data 
import numpy as np 
import sklearn.linear_model
import sklearn.naive_bayes 
import sklearn.cross_validation
import sklearn.ensemble
import sklearn.decomposition

                
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
		print "Months to live = %d, n_dead = %d, ROC AUC = %0.4f" % (i, np.sum(Y), auc)


def error(Y_pred, Y_true):
	return np.mean(np.abs(Y_pred - Y_true))

def average_expert_error(experts, Y):
	Y_expert_combined = np.zeros_like(Y)
	Y_expert_count = np.zeros_like(Y, dtype=int)

	for expert in experts:
		Y_pred = experts[expert]
		mask = np.array(~(Y_pred.isnull()))
		print expert.strip(), "n =", np.sum(mask)
		Y_pred_subset = np.array(Y_pred[mask].astype('float'))
		print "-- %0.4f" % error(Y_pred_subset, Y[np.array(mask)])
		Y_expert_combined[mask] += Y_pred_subset
		Y_expert_count[mask] += 1

	combined_mask = Y_expert_count > 0
	Y_expert_combined = Y_expert_combined[combined_mask]
	Y_expert_combined /= Y_expert_count[combined_mask]
	return error(Y_expert_combined, Y[combined_mask])

if __name__ == '__main__':

	X, Y, dead, experts, test_set_mask = data.load_dataset(binarize_categorical = True, greedy_feature_binning = False)
	X = np.array(X)

	
	print "Data shape", X.shape

	print "---"
	print "Average prediction error = %0.4f" % average_expert_error(experts, Y)
	print "---"

	n_test = np.sum(test_set_mask)
	n_train = len(Y) - n_test
	print "Training set = %d, test set = %d" % (n_train, n_test)

	X_train = X[~test_set_mask]
	Y_train = Y[~test_set_mask]

	X_test = X[test_set_mask]
	Y_test = Y[test_set_mask]

	pca = sklearn.decomposition.PCA(10)
	X_train = pca.fit_transform(X_train)
	print "PCA transformed shape", X_train.shape
	X_test = pca.transform(X_test)

	def fit(model):
		model.fit(X_train, Y_train)
		pred = model.predict(X_test)
		e = error(pred, Y_test)
		print "%s error: %0.4f" % (model.__class__.__name__, e)

	fit(sklearn.linear_model.LinearRegression())
	fit(sklearn.svm.SVR())
	fit(sklearn.linear_model.RidgeCV(alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]))
	fit(sklearn.linear_model.LassoCV())
	fit(sklearn.linear_model.OrthogonalMatchingPursuitCV())
	fit(sklearn.ensemble.RandomForestRegressor(n_estimators = 500))


