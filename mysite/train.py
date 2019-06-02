import pickle
import csv
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import classify


def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name

    class Data:
        pass

    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar, trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")
    from sklearn.feature_extraction.text import TfidfVectorizer
    sentiment.tfidf_vect = TfidfVectorizer()
    sentiment.trainX = sentiment.tfidf_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.tfidf_vect.transform(sentiment.dev_data)
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment


def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label, text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels


def preprocess(filename):
    columns = defaultdict(list)
    with open(filename, encoding="utf8") as csvfile:
        readCSV = csv.DictReader(csvfile)
        for row in readCSV:
            for (k,v) in row.items():
                columns[k].append(v)
    return columns


def generate_data(columns, tag):
    toxic_types = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    data = columns["comment_text"]
    toxic = np.array([columns[feature] for feature in toxic_types],dtype=int).T
    Y = list(np.apply_along_axis(lambda l: np.sum(l), 1, toxic))
    positive = [(d, 1) for d, y in zip(data, Y) if y > 0]
    negative = [(d, y) for d, y in zip(data, Y) if y == 0]
    data, Y = tuple(zip(*(positive+negative)))
    data = list(data)
    Y = list(Y)
    return data, Y


def train_part1_model():
	print("Reading data")
	tarfname = "data/sentiment.tar.gz"
	sentiment = read_files(tarfname)
	print("\nTraining classifier")
	cls = classify.train_classifier(sentiment.trainX, sentiment.trainy)
	save_model({'part1_vect.pk': sentiment.tfidf_vect, 'part1_model.pk': cls})


def train_part2_model():
    columns_train = preprocess("train.csv")
    data_train, y_train = generate_data(columns_train, 'Train')
    columns_test = preprocess("test.csv")
    columns_test_labels = preprocess("test_labels.csv")
    columns_test_labels["comment_text"] = columns_test["comment_text"]
    data_test, y_test = generate_data(columns_test_labels, 'Test')
    data = data_train + data_test
    tfidf_vect = TfidfVectorizer()
    tfidf_vect.fit(data)
    X_train = tfidf_vect.transform(data_train)
    cls = classify.train_classifier(X_train, y_train)
    save_model({'part2_vect.pk': tfidf_vect, 'part2_model.pk': cls})


def save_model(models):
    for file_name, model in models.items():
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)


def load_model(file_name):
    with open(file_name, 'rb') as f:
        model = pickle.load(f)
    return model


if __name__ == '__main__':
	train_part1_model()
	train_part2_model()

