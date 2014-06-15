import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import ast
import random

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
    
def nums_in_title (df, title_col):
    title_words = [s.split() for s in df[title_col]]
    num_list = []
    for title in title_words:
        title_nums = [int(word) for word in title if word.isdigit()]
        num_list.append(any([n for n in title_nums]))
    return num_list

def find_tags (body):
    spaces, comma_ind, space_ind = 0, 0, 0
    tags = []
    for index, ch in enumerate(reversed(body)):
        if ch == ',':
            spaces = 0
            tags.insert(0,body[(index * -1):((comma_ind + 1) * -1 if comma_ind != 0 else None)].strip())
            comma_ind = index   
        elif ch == ' ':
            spaces += 1
            if spaces == 1:
                space_ind = index
            elif spaces > 4:
                if comma_ind != 0:
                    tags.insert(0,body[(space_ind * -1):((comma_ind + 1) * -1)].strip())
                return tags
    return tags  
    
def find_all_tags (df, body_col):
    tag_list = []
    has_tags = []
    for body in df[body_col]:
        if len(body) > 0 and body[-1] == ',':
            body = body[:(len(body) - 1)]
        tags = find_tags (body)
        tag_list.append([tag for tag in tags if tag != ''])
        has_tags.append(False if tags == [] else True)
    return tag_list, has_tags              

def tag_split (tag_list, has_tags, tag_labels, train_frac):
    only_tags = [(i, tags) for (i, tags) in enumerate(tag_list) if has_tags[i] == True]
    
    only_tag_labels = [tag_labels for (i, tag_labels) in enumerate(su_df['label']) if has_tags[i] == True]    
    
    feature_train_list = []
    feature_test_list = []
    label_train = []
    label_test = []
    for ii in xrange(len(only_tags)):
        r = random.random()
        if r > train_frac:
            feature_test_list.append(only_tags[ii])
            label_test.append(only_tag_labels[ii])
        else:
            feature_train_list.append(only_tags[ii])
            label_train.append(only_tag_labels[ii])
    return feature_train_list, feature_test_list, label_train, label_test 
    
def make_feature_dicts (feature_list):
    feature_dicts = []
    for (i,tags) in feature_list:
        feature_dict = {}
        for tag in tags:
            feature_dict[tag] = 1
        feature_dicts.append(feature_dict)
    return feature_dicts
    
def vectorize_tags (feature_train_dicts, feature_test_dicts = None):
    dv1 = DictVectorizer()
    feature_train = dv1.fit_transform(feature_train_dicts)
    if feature_test_dicts:
        feature_test = dv1.transform(feature_test_dicts)
        return feature_train, feature_test
    return feature_train
    
def tag_naive_bayes (df, body_col, label_col, train_frac = 0.8):
    tag_list, has_tags = find_all_tags(df, body_col)
    df['has_tags'] = has_tags
    su_df['num_of_tags'] = [len(tags) for tags in tag_list]
    feature_train_list, feature_test_list, label_train, label_test = tag_split (tag_list, has_tags, df[label_col], train_frac)
    feature_train, feature_test = vectorize_tags (make_feature_dicts (feature_train_list), make_feature_dicts (feature_test_list))
    
    mnb = MultinomialNB()
    mnb.fit(feature_train, label_train)
    return mnb.predict_proba(feature_test), mnb.score(feature_test, label_test)   

def is_recipe (df, body_col, recipe_list = None):
    if not recipe_list:
        recipe_list = ['recipe', 'cup', 'tablespoon', 'teaspoon', 'tbsp', 'tsp', 'salt', 'pepper', 
                       'butter', 'oil', 'flour', 'egg', 'onion', 'garlic']
    is_recipe = []
    for row in df[body_col]:
        for item in recipe_list:
            if item in row:
                is_recipe.append(True)
                break
        else:
            is_recipe.append(False)
    return is_recipe

def is_sweet (df, body_col):
    is_sweet = []
    for row in su_df[body_col]:
        if 'sugar' in row:
            is_sweet.append(True)
        else:
            is_sweet.append(False) 
    return is_sweet 

def topWords(centers, vocab):
    for i, center in enumerate(centers):
        print 'Subpopulation ' + str(i) + ':'
        for j in center.argsort()[-20:][::-1]:
            print '\t' + vocab[j]
        print '\n'

def kmeansError (features, km):
    error = 0
    for i in range(len(km.labels_)):
        error += (features[i] - km.cluster_centers_[km.labels_[i]]) ** 2
    return error
        

def findElbow(features, n = 10):
    error = []
    for i in xrange(n):
        km = KMeans(n_clusters = i + 1)
        km.fit_transform(features)
        error.append(kmeansError(features, km))
    plt.figure(figsize=(10,10))
    plt.plot(range(1,n + 1),error,'k',linewidth=10)
    plt.plot(range(1,n + 1),error,'ko',markersize=25)
    plt.show()

if __name__ == "__main__":
    su_df = pd.read_table('train.tsv')
    
    su_df['title'], su_df['body'], su_df['url_text'] = convert_text(su_df, 'boilerplate')
    
    su_df['nums'] = nums_in_title(su_df, 'title')

    su_df['is_recipe'] = is_recipe (su_df, 'body')
    su_df['is_sweet'] = is_sweet (su_df, 'body')
    
    dv = DictVectorizer()
    tag_list, has_tags = find_all_tags(su_df, 'body')
    only_tag_list = [(i, tags) for (i, tags) in enumerate(tag_list) if has_tags[i] == True]
    feature_dicts = make_feature_dicts (only_tag_list)
    features = dv.fit_transform(feature_dicts)
    vocab = dv.get_feature_names()
    
    findElbow(features, 20)
    
    # kt = KMeans(n_clusters = 25)    
    # kt.fit_transform(features)
    # labels = kt.labels_
    # centers = kt.cluster_centers_
    
    
    # print topWords(centers, vocab)
    

