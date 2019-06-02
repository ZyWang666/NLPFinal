from django.shortcuts import render

from .models import Comment
from .models import MyModel

from lime import lime_text
from lime.lime_text import LimeTextExplainer


### coefficient of model


def index(request):
	comment_list = Comment.objects.order_by('comment_text')
	return render(request,  'comments/index.html', {'comment_list': comment_list})


def results(request):
	comment = request.POST['fname']
	label = request.POST['flabel']
	n_features = request.POST['ffeatures']
	passed_model = request.POST['fmodel']
	if passed_model == 'Sentiment':
		model = MyModel.part1_model
	else:
		model = MyModel.part2_model
	class_names = ['negative', 'positive']
	explainer = LimeTextExplainer(class_names=class_names)
	exp = explainer.explain_instance(comment, model.predict_proba, num_features=int(n_features))

	prediction = class_names[0] if model.predict([comment]) == 0 else class_names[1]
	features = exp.as_list()
	pos_pro, neg_pro = exp.predict_proba
	s_comment = comment.split(' ')
	words = [x for x, y in features]
	p = round(pos_pro,2)
	n = round(neg_pro,2)
	pos_pro *= 1000
	neg_pro *= 1000
	return render(request, 'comments/results.html', {
		"prediction": prediction,
		'features': features,
		's_comment': s_comment,
		'words': words,
		'label':label,
		'pos_pro': pos_pro,
		'neg_pro':neg_pro,
		'p': p,
		'n': n}
	)



