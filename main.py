#get data from csv
import csv
from collections import defaultdict
columns = defaultdict(list)
with open('GrammarandProductReviews.csv') as csvfile:
	readCSV = csv.DictReader(csvfile)
	for row in readCSV:
		for (k,v) in row.items():
			columns[k].append(v)

#columns['reviews.rating'] = rating
#columns['reviews.text'] = text