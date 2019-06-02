from django.db import models

from sklearn.pipeline import make_pipeline
from train import load_model


class Comment(models.Model):
    comment_text = models.CharField(max_length=200)


class MyModel(models.Model):
	vectorizer1 = load_model('part1_vect.pk')
	model1 = load_model('part1_model.pk')
	part1_model = make_pipeline(vectorizer1, model1)
	vectorizer2 = load_model('part2_vect.pk')
	model2 = load_model('part2_model.pk')
	part2_model = make_pipeline(vectorizer2, model2)


