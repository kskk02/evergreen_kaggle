from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as p
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

def getDender(bags, titles, figsize=(40,15)):
    distxy = squareform(pdist(bags, metric='euclidean'))

    plt.figure(figsize=figsize)
    R = dendrogram(linkage(distxy, method='complete'), orientation = 'right', labels=titles)

    xlabel('distance')
    ylabel('headlines')
    suptitle('Cluster Dendrogram', fontweight='bold', fontsize=14)

def convert_text (df, col):
    title_list = []
    body_list = []
    url_list = []
    for (i, row) in enumerate(df[col]):
        try:
            bp_dict = ast.literal_eval(row)
        except:
            bp_dict = {}
        for k,v in bp_dict.iteritems():
            if k == 'title':
                title_list.append(v)
            if k == 'body':
                body_list.append(v)
            if k == 'url':
                url_list.append(v)
        if len(title_list) == i:
            title_list.append('')
        if len(body_list) == i:
            body_list.append('')
        if len(url_list) == i:
            url_list.append('') 
    return title_list, body_list, url_list 

def topWords(centers, vocab):
    for i, center in enumerate(centers):
        print 'Subpopulation ' + str(i) + ':'
        for j in center.argsort()[-20:][::-1]:
            print '\t' + vocab[j]
        print '\n'

def main():

	traindata = (p.read_table('train.tsv'))
	tr_title, tr_body, tr_url = convert_text(traindata)

	testdata = list(np.array(p.read_table('test.tsv'))[:,2])
	y = np.array(p.read_table('train.tsv'))[:,-1]

	wordCount = cv(stop_words = 'english', encoding='latin-1')
	wordTFIDF = tfidf(stop_words = 'english', encoding='latin-1')

	corpus = tr_body

	bag = wordCount.fit_transform(corpus)
	tfdif = wordTFIDF.fit_transform(corpus)

	tfdif = tfdif.toarray()

	kmeans_soln.getDender(bag, tr_title)

	titles = np.array(tr_title)

	vocab = wordCount.get_feature_names()
	vocabTF = wordTFIDF.get_feature_names()

	topWords(centers, vocab)

if __name__=="__main__":
	main()