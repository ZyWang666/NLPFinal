from django.shortcuts import render

from .models import Comment
from .models import MyModel

from lime import lime_text
from lime.lime_text import LimeTextExplainer

import string
import numpy as np

### coefficient of model


def index(request):
	comment_list = Comment.objects.order_by('comment_text')
	return render(request,  'comments/index.html', {'comment_list': comment_list})


def results(request):
	# text
	comment = request.POST['fname']
	# label	
	label = request.POST['flabel']
	# number of features	
	n_features = request.POST['ffeatures']
	# selected model	
	passed_model = request.POST['fmodel']
	# select model	
	if passed_model == 'Sentiment':
		fns = MyModel.vectorizer1.get_feature_names()
		cls = MyModel.model1
		model = MyModel.part1_model
		pos_pre = 'positive review'
		neg_pre = 'negative review'
	else:
		fns = MyModel.vectorizer2.get_feature_names()
		cls = MyModel.model2
		model = MyModel.part2_model
		pos_pre = 'toxic comment'
		neg_pre = 'non-toxic comment'
	# class names of lime	
	class_names = ['negative', 'positive']
	# construct lime text explainer	
	explainer = LimeTextExplainer(class_names=class_names)
	# explain the instance passed in	
	exp = explainer.explain_instance(comment, model.predict_proba, num_features=int(n_features))
	# prediction of model
	prediction = class_names[0] if model.predict([comment]) == 0 else class_names[1]
	if prediction == 'positive':
		prediction = pos_pre
	else:
		prediction = neg_pre
	if label == 'positive':
		label = pos_pre
	else:
		label = neg_pre
	# features of lime	
	features = exp.as_list()
	features = [(p[0],round(p[1],2)) for p in features]
	# positive probability and negative probability of model	
	neg_pro, pos_pro = exp.predict_proba
	# striped and splited text
	s_comment = comment.translate(str.maketrans('', '', string.punctuation)).split(' ')
	t_features = top_features(s_comment, fns, cls, int(n_features))
	t_features = sorted(t_features, key=lambda x: abs(x[1]), reverse=True)
	print(t_features)
	words = [x for x, y in features]
	# rounded positive probability	
	p = round(pos_pro, 2)
	# rounded negative probability
	n = round(neg_pro, 2)
	pos_pro *= 1000
	neg_pro *= 1000
	lime_highest_value = abs(features[0][1])
	model_highest_value = abs(t_features[0][1])
	#length = max_value/(max_leng-min_leng)*value+min_leng
	new_f = []
	for pair in features:
		new_f.append((pair[0], (500/lime_highest_value) * pair[1]))
	features = new_f
	new_t = []
	for pair in t_features:
		new_t.append((pair[0], (500/model_highest_value) * pair[1]))
	t_features = new_t
	return render(request, 'comments/results.html', {
		'prediction': prediction,
		'features': features,
		's_comment': s_comment,
		'words': words,
		'label':label,
		'pos_pro': pos_pro,
		'neg_pro':neg_pro,
		'p': p,
		'n': n,
		't_features':t_features,
		'pos_pre': pos_pre,
		'neg_pre': neg_pre}
	)


def top_features(data, fns, cls, k=6):
    features = list(set(data))
    coefficients = cls.coef_[0]
    coefs = np.array([coefficients[fns.index(feature)] if feature in fns else 0 for feature in features])
    k = min(k, len(coefs))
    abs_coefs = np.abs(coefs)
    positive_indices = np.argpartition(abs_coefs, -k)[-k:]
    positive_coefs = abs_coefs[positive_indices]
    positive_indices = positive_indices[np.argsort(-positive_coefs)]
    positive_coefs = coefs[positive_indices]
    features = np.array(features)
    positive_indices = [int(i) for i in positive_indices]
    positive_features = features[positive_indices]
    return list(zip(positive_features, positive_coefs))

