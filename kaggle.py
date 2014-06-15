import numpy as np
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as p
from pprint import pprint
from time import time
import logging
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import cross_validation
import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans, MiniBatchKMeans
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from sklearn import metrics
from sklearn.metrics import mean_squared_error
# general strategy -
# Assumptions:
# 1. Text is the only valuable info that defines evergreen.

# strategy
# 1. Break up website content to various high level features -e.g. body, title, keywords
# 2. featurize these high level features (start with just CountVectorizer, next try to categorize using clustering and subjectively observe keywords to derive a more valuable categorization)
# 3. using LogisticRegression for each individual high level feature

def IsEnglish (word):
    # this function uses synsets (english module) to find out whether the word that is passed in is in the english languge or not.
  if not wordnet.synsets(word):
    return False
  else:
    return True

def StringCleanUp (text):
# this function takes an entire review (multiple sentences) and returns back a modified sentence that only contains words (alphabets)
# and english language words only
  if text == None:
    return " "
  stemmer = PorterStemmer()
  alphabetic_only = "".join(c for c in text if c == " " or c.isalpha())
  split_text = alphabetic_only.split()
  res = [stemmer.stem(tex) for tex in split_text if IsEnglish(tex)]
  restored_text = " ".join(res)
  return restored_text


def top10words_cluster (df,column,num_of_clusters) :

    input_vectorizer = TfidfVectorizer(stop_words='english')
    X_train = input_vectorizer.fit_transform(df[column])
    vocab = input_vectorizer.get_feature_names()

    sse_err = []
    k=num_of_clusters
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
    res = km.fit(X_train)
    x = X_train.toarray()
    SSE = mean_squared_error(res.cluster_centers_[res.labels_],x)
    sse_err.append(SSE)

    #plt.plot(range(1,8),sse_err)
    #plt.show()

    vocab = np.array(vocab)
    cluster_centers = np.array(res.cluster_centers_)
    sorted_vals = [res.cluster_centers_[i].argsort() for i in range(0,np.shape(res.cluster_centers_)[0])]

    threshold = 0.01
    for i in xrange(len(res.cluster_centers_)):
        words=vocab[sorted_vals[i][-10:]]
        print "\n For Centroid ",i," keywords are ", words


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

df = pd.read_table('data/train.tsv')
data = df.boilerplate[0:50]
data = data.apply(json.loads)   
label = df.iloc[:,26]
    
keywords  = [StringCleanUp(data[i]["url"])  if "url" in data[i] else " " for i in range(0,len(data)) ]
bodies  = [StringCleanUp(data[i]["body"])  if "url" in data[i] else " " for i in range(0,len(data)) ]
titles  = [StringCleanUp(data[i]["title"])  if "url" in data[i] else " " for i in range(0,len(data)) ]
data=pd.DataFrame(data)
data["bodies"]=bodies
data["keywords"]=keywords
data["titles"]=titles
data["label"] = label


alc_cat = pd.get_dummies(df.alchemy_category)

full_data = pd.merge(df,alc_cat,right_index=True,left_index=True)
correlations = full_data.corr()
correlations.label

top10words_cluster(data[data["bodies"]!=" "],"bodies",10)

###############################################################################
# define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression()),
])

parameters = {
    'vect__max_df': (0.5 ),#, 0.75, 1.0),
    'vect__max_features': (None), #, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
  #  'tfidf__norm': ('l1', 'l2'),
 #   'clf__alpha': (0.0001,0.00001, 0.000001),
#    'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
}


traindata = data.keywords
testdata = list(np.array(p.read_table('./data/test.tsv'))[:,2])
y = np.array(p.read_table('./data/train.tsv'))[:,-1]

tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
    analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)

rd = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                         C=1, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight=None, random_state=None)

X_all = traindata + testdata
lentrain = len(traindata)

print "fitting pipeline"
tfv.fit(X_all)
print "transforming data"
X_all = tfv.transform(X_all)

X = X_all[:lentrain]
X_test = X_all[lentrain:]



xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(extract, label, test_size=0.15, random_state=42)

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring="roc_auc")

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()
grid_search.fit(xtrain, ytrain)
print("done in %0.3fs" % (time() - t0))
print()



print "20 Fold CV Score: ", np.mean(cross_validation.cross_val_score(rd, X, y, cv=20, scoring='roc_auc'))

print "training on full data"
rd.fit(X,y)
pred = rd.predict_proba(X_test)[:,1]
testfile = p.read_csv('./data/test.tsv', sep="\t", na_values=['?'], index_col=1)
pred_df = p.DataFrame(pred, index=testfile.index, columns=['label'])
pred_df.to_csv('./data/benchmark.csv')
print "submission file created.."



print correlations.label





