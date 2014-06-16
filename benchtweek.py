"""
beating the benchmark @StumbleUpon Evergreen Challenge
__author__ : Abhishek Thakur
"""

# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as p
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline


loadData = lambda f: np.genfromtxt(open(f,'r'), delimiter=' ')

def clusterOutcome(labels, y):
    ys = defaultdict
    avgs = defaultdict

    for i, c in enumerate(labels):
        ys[c].append(y[i])
    
    for x in unique(labels):
        avgs[x].append(np.mean(ys[x]))
        
    return ys, avgs

def topWords(centers, vocab):
    for i, center in enumerate(centers):
        print 'Subpopulation ' + str(i) + ':'
        for j in center.argsort()[-20:][::-1]:
            print '\t' + vocab[j]
        print '\n'

def extract_domain(url):
    # extract domains
    domain = url.lower().split('/')[2]
    domain_parts = domain.split('.')

    # e.g. co.uk
    if domain_parts[-2] not in ['com', 'co']:
        return '.'.join(domain_parts[-2:])
    else:
        return '.'.join(domain_parts[-3:])

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

def getURL(data):
  urls = []
  for x in data:
      urls.append(extract_domain(x))

  urlDum = p.get_dummies(urls)
  url = np.array(urlDum)*10
  return url

def clusters(corpus, lentrain):
  wordCount = cv(stop_words = stopwords, ngram_range= (1,2), encoding='latin-1')
  bag = wordCount.fit_transform(corpus)
  vocab = wordCount.get_feature_names()
  kt = cluster.KMeans(n_clusters = 5)
  count = kt.fit_transform(bag)
  labels = kt.labels_
  centers = kt.cluster_centers_

  lentrain = len(tr_body)
  tmpLabels = labels[:lentrain]
  topWords(centers, vocab)
  outcomeCnt, outcomeAvg = clusterOutcome(tmpLabels, y_train)

def main():

  print "loading data.."
  traindata = (p.read_table('train.tsv'))
  tr_title, tr_body, tr_url = convert_text(traindata,'boilerplate')

  testdata = p.read_table('test.tsv')
  ts_title, ts_body, ts_url = convert_text(testdata,'boilerplate')

  y = np.array(p.read_table('train.tsv'))[:,-1]

  internetStopWords = ['http', 'www', 'online', 'com', 'jpg', 'static', 'link', 'terminal01', 'user', 'null', 'div', 'span', 'font', 'timestamp', 'content', 'blog']
  stopwords = ENGLISH_STOP_WORDS
  stopwords = list(stopwords)
  stopwords = stopwords + internetStopWords

  X_all = tr_body + ts_body + tr_title + ts_title

  #use for dummy variables 
  urls = getURL(traindata['url'])
  
  #building the model 
  tfv = TfidfVectorizer(min_df=3, stop_words = stopwords, max_features=None, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1)
  tfdif = wordTFIDF.fit_transform(corpus)
  Xt = tfdif[:lentrain]

  rnd = lm.RandomizedLogisticRegression()
  xrnd = rnd.fit_transform(Xt, y_train)

  # not working : 
  #X_all = hstack( (xrnd,url) )
  # tfv.build_analyzer()
  rd = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=None)

  lentrain = len(traindata)

  print "fitting pipeline"
  tfv.fit(X_all)
  print "transforming data"
  X_all = tfv.transform(X_all)

  X = X_all[:lentrain]
  X_test = X_all[lentrain:]



  print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(rd, X, y, cv=20, scoring='roc_auc'))

  # print "training on full data"
  # rd.fit(X,y)
  # pred = rd.predict_proba(X_test)[:,1]
  # testfile = p.read_csv('test.tsv', sep="\t", na_values=['?'], index_col=1)
  # pred_df = p.DataFrame(pred, index=testfile.index, columns=['label'])
  # pred_df.to_csv('benchmark.csv')
  # print "submission file created.."

if __name__=="__main__":
  main()
