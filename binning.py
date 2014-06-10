
def cutoff_tuples_recursive(values, cutoffs, n_cutoffs):
	
	if n_cutoffs < 1 or len(values) == 0:
		return [cutoffs]
	results = []
	for v in values:
		remaining_values = [v2 for v2 in values if v2 > v]
		results.extend(cutoff_tuples_recursive(remaining_values, cutoffs + (v,), n_cutoffs - 1))
	return results

def cutoff_tuples(values):
	values = list(sorted(np.unique(values)))
	if len(values) == 2:
		return (values[0],)
	if len(values) > 10:
		_, values = np.histogram(values)
		values = values[1:]
	# drop the maximum value, since we're looking for features greater than this value
	values = values[:-1]
	results = []
	for i in xrange(1, min(len(values)+1, 4)):

		results.extend(cutoff_tuples_recursive(values, (), i+1))
	return results 

def encoding_score(encoded, n, y_train):
	group_scores = []
	total = float(len(y_train))

	best_survival = 0 
	best_survival_size = 0
	worst_survival = np.inf
	worst_survival_size = 0  
	for i in xrange(n):	
		mask = (encoded == i)
		n_mask = mask.sum()
		y_train_subset = y_train[mask]
		average = np.mean(y_train_subset)
		if average > best_survival:
			best_survival = average
			best_survival_size = n_mask 
		if average < worst_survival:
			worst_survival = average 
			worst_survival_size = n_mask 
	return (np.abs(best_survival - worst_survival) * (best_survival_size + worst_survival_size)) / total 

def best_binning_encoding(df, field_name, y, train_mask):
	x = df[field_name]
	x = np.array(x).squeeze()
	y = np.array(y).squeeze()
	x_train = x[train_mask]
	y_train = y[train_mask]

	cutoffs = cutoff_tuples(x_train)
	best_score = 0
	best_cutoff = None 
	best_encoded = None 
 
	for cutoff in cutoffs:
		n = len(cutoff)
		encoded = np.searchsorted(cutoff, x, side = 'right')
		encoded_train = encoded[train_mask]
		score = encoding_score(encoded_train, n, y_train)
	
		if score > best_score:
			best_cutoff = cutoff 
			best_score = score 
			best_encoded = encoded
	print "Best thresholds for %s = %s" % (field_name, best_cutoff)
	return best_encoded


def best_single_threshold_encoding(df, field_name, y, train_mask):
	x = df[field_name]
	x = np.array(x).squeeze()
	y = np.array(y).squeeze()
	x_train = x[train_mask]
	y_train = y[train_mask]

	best_score = np.inf 
	best_cutoff = None 
	best_encoded = None 
 
	for threshold in list(sorted(np.unique(x_train)))[:-1]:
		right_mask = x_train > threshold
		left_mask = ~right_mask 
		n_right = right_mask.sum()
		n_left = left_mask.sum()
		score = np.var(y_train[right_mask]) * n_right + np.var(left_mask) * n_left 
	
		if score < best_score:
			best_cutoff = threshold 
			best_score = score 
			best_encoded = x > threshold
	print "Best threshold for %s = %s" % (field_name, best_cutoff)
	return best_encoded
