#get data from csv
import csv
import classify
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict

def getMyModel():
	columns = defaultdict(list)

	if __name__ == "__main__":
		with open('GrammarandProductReviews.csv', encoding="utf8") as csvfile:
			readCSV = csv.DictReader(csvfile)
			for row in readCSV:
				for (k,v) in row.items():
					columns[k].append(v)

		# Preprocess data, get rid of reviews that don't have recommendation
		features = list(columns.keys())
		print(features)
		data = columns["reviews.text"]
		Y = columns["reviews.doRecommend"]
		XY = zip(data,Y)
		X, Y = [], []
		for x,y in XY:
			if y == 'TRUE':
				X.append(x)
				Y.append(1)
			elif y == 'FALSE':
				X.append(x)
				Y.append(0)

		# The first 90% are train data, the last 10% are test data
		n = int(len(X)*.9)
		XY_train = list(zip(X,Y))[:n]
		XY_test = list(zip(X,Y))[n:]
		data_train, y_train = [x for x, y in XY_train], [y for x, y in XY_train]
		data_test, y_test = [x for x, y in XY_test], [y for x, y in XY_test]
		print("Train data has %d positive reviews"%y_train.count(1))
		print("Train data has %d negative reviews"%y_train.count(0))
		print("Test data has %d positive reviews"%y_test.count(1))
		print("Test data has %d negative reviews"%y_test.count(0))

		# Testing
		print("Testing CountVectorizer...")
		count_vect = CountVectorizer()
		count_vect.fit(data)
		X_train = count_vect.transform(data_train)
		X_test = count_vect.transform(data_test)
		cls = classify.train_classifier(X_train, y_train)
		classify.evaluate(X_test, y_test, cls, 'test')

		print("Testing TfidfVectorizer...")
		tfidf_vect = TfidfVectorizer()
		tfidf_vect.fit(data)
		X_train = tfidf_vect.transform(data_train)
		X_test = tfidf_vect.transform(data_test)
		cls = classify.train_classifier(X_train, y_train)
		classify.evaluate(X_test, y_test, cls, 'test')

		#lime
		from lime import lime_text
		from sklearn.pipeline import make_pipeline
		c = make_pipeline(tfidf_vect, cls)
		print(c.predict_proba([data_test[0]]))

		from lime.lime_text import LimeTextExplainer
		class_names = ['recommend', 'notRecommend']
		explainer = LimeTextExplainer(class_names=class_names)
		idx = 5001
		exp = explainer.explain_instance(data_test[idx], c.predict_proba, num_features=6)
		print('Document id: %d' % idx)
		print('Probability(not recommend) =', c.predict_proba([data_test[idx]])[0,1])
		print('True class: %s' % class_names[y_test[idx]])

		exp.as_list()
		return cls









