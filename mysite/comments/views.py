from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.http import Http404

from .models import Comment
from .models import MyModel

# Create your views here.


### probability
### prediction
### highlight
### coefficient of model
### weight of lime

def index(request):
	comment_list = Comment.objects.order_by('comment_text')
	return render(request,  'comments/index.html', {'comment_list': comment_list})

def results(request):
	comment = request.POST['fname']
	label = request.POST['flabel']
	model = MyModel.res

	from lime import lime_text

	from lime.lime_text import LimeTextExplainer
	class_names = ['recommend', 'notRecommend']
	explainer = LimeTextExplainer(class_names=class_names)
	exp = explainer.explain_instance(comment, model.predict_proba, num_features=3)

	prediction = class_names[0] if model.predict([comment]) == 0 else class_names[1]
	features = exp.as_list()
	pos_pro, neg_pro = exp.predict_proba

	s_comment = comment.split(' ')

	words = [x for x, y in features]
	return render(request, 'comments/results.html', {
		"prediction": prediction,
		'features': features,
		's_comment': s_comment,
		'words': words,
		'label':label,
		'pos_pro': pos_pro,
		'neg_pro':neg_pro}
	)
