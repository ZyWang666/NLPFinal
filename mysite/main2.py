import numpy as np

def test():
	print('calling test')
	return 'a'


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

def display_features(fns, cls):
    from tabulate import tabulate
    coefs = cls.coef_[0]
    positive_indices = np.argpartition(coefs, -10)[-10:]
    positive_coefs = coefs[positive_indices]
    positive_indices = positive_indices[np.argsort(-positive_coefs)]
    positive_coefs = coefs[positive_indices]
    positive_features = fns[positive_indices]
    negative_indices = np.argpartition(coefs, 10)[:10]
    negative_coefs = coefs[negative_indices]
    negative_indices = negative_indices[np.argsort(negative_coefs)]
    negative_coefs = coefs[negative_indices]
    negative_features = fns[negative_indices]
    header = ["Feature name", "Coefficient"]
    positive_data = [(positive_features[i], round(positive_coefs[i],1)) for i in range(10)]
    negative_data = [(negative_features[i], round(negative_coefs[i],1)) for i in range(10)]
    positive_table = tabulate(positive_data, headers=header)
    negative_table = tabulate(negative_data, headers=header)
    print(positive_table)
    print(negative_table)
    positive_table = tabulate(positive_data, headers=header, tablefmt='latex')
    negative_table = tabulate(negative_data, headers=header, tablefmt='latex')
    return positive_table, negative_table

def getMyModel():
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    print("\nTraining classifier")
    import classify
    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy)

    fns = sentiment.tfidf_vect.get_feature_names()
    fns = np.array(fns)
    display_features(fns, cls)
getMyModel()






