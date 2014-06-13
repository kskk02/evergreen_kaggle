
# coding: utf-8

# In[1]:

get_ipython().magic(u'pylab inline')
import numpy as np
import random
from sklearn import cluster, datasets 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[2]:

iris = datasets.load_iris()
x_iris = iris.data
y_iris = iris.target

scale = StandardScaler(copy=True)
scale.fit_transform(x_iris);


# In[3]:

def getdistance(x, y):
    return np.sqrt(sum((x-y)**2))


# In[4]:

# randomly sampled guesses

#def initguesses(x,k):
#    # returns list of k guesses
#    return random.sample(x,k)


# In[5]:

# weighted guesses

def weighted_choice(weights):
    totals = []
    running_total = 0

    for w in weights:
        running_total += w
        totals.append(running_total)

    rnd = random.random() * running_total
    for i, total in enumerate(totals):
        if rnd < total:
            return i
        
def initguesses(x,k):
    centers = []
    weights = np.ones((x.shape[0],))
    choice = weighted_choice(weights)
    centers.append(x[choice]) #first center
    
    for i in xrange(k-1):
        for j,pt in enumerate(x):
            last_center = centers[-1]
            weights[j] *= getdistance(pt,last_center)
        choice = weighted_choice(weights)
        centers.append(x[choice]) #next center
    return centers


# In[6]:

def findnearest(x,centers):
    nearestpts = []
    labels = []
    for i in xrange(len(centers)):
        nearestpts.append(np.array([]))
    
    count = np.zeros((len(centers),))
    
    for pt in x:
        tmpnearest = 9999
        tmpi = 9999
        for i,c in enumerate(centers):
            dist = getdistance(c,pt)
            if dist < tmpnearest:
                tmpnearest = dist
                tmpi = i
        if tmpnearest == 9999: print 'uhoh'
        else:
            if count[tmpi] == 0:
                nearestpts[tmpi] = pt
            else:
                nearestpts[tmpi] = np.vstack((nearestpts[tmpi],pt))
            labels.append(tmpi)
            count[tmpi] += 1
        
    #returns list of k-arrays of nearest points
    return labels,nearestpts


# In[7]:

def getCentroid(x):
    return 1.0*np.sum(x,axis=0)/x.shape[0]


# In[92]:

centers = initguesses(x_iris, 3) 
centers


# In[9]:

labels,nearest = findnearest(x_iris, centers)
print len(nearest[0])
print len(nearest[1])


# In[10]:

x = np.array([[1,4,-5],[3,-2,9]])
c = getCentroid(x)
print c


# In[32]:

def kmeans(x,n):
    centers = initguesses(x, n)
    centerpair = []
    for i in xrange(20):
        labels,nearests = findnearest(x, centers)
        for j,nearest in enumerate(nearests):
            #print nearest
            c = getCentroid(nearest)
            centers[j] = c
    labels,nearests = findnearest(x, centers)

    return labels, nearests, centers


# In[33]:

labels, nearests, centers = kmeans(x_iris,3)


# In[12]:

def rmsq(pts,centers):
    sumdist = 0.0
    for i,center in enumerate(centers):
        for pt in pts[i]:
            sumdist += (1.0*getdistance(center,pt))**2
    return sumdist


# In[13]:

error = rmsq(nearests,centers)


# In[14]:

from collections import Counter
cnt = Counter()
for l in labels:
    cnt[l] +=1
print cnt


# In[15]:

k_means = cluster.KMeans(n_clusters=3)
k_means.fit(x_iris) 
print k_means.labels_
# print k_means.cluster_centers_


# In[16]:

cnt = Counter()
for l in y_iris:
    cnt[l] +=1
print cnt


# In[39]:

fig = plt.figure(figsize=(15,15));

ax = fig.add_subplot(111, projection='3d')
xs = x_iris[:,0]
ys = x_iris[:,1]
zs = x_iris[:,2]
cs = x_iris[:,3]
ax.scatter(xs, ys, zs, c=cs)

#plt.plot(x_iris[:,2],x_iris[:,3],'bo');
#for i in xrange(4):
#    plt.plot(centers[i][2],centers[i][3],'go',markersize=10,linewidth=0)
#    plt.plot(k_means.cluster_centers_[i][2],k_means.cluster_centers_[i][3],'ro',markersize=10)
cmap = {0: 'red', 1: 'blue', 2: 'green'}
for i in xrange(3):
    c = cmap[i]
    xs = centers[i][0]
    ys = centers[i][1]
    zs = centers[i][2]
    ax.scatter(xs, ys, zs, color=c, s=100, linewidth=3)

for label, nearest in enumerate(nearests):
    for pt in nearest:
        c = cmap[label]
        xs = pt[0]
        ys = pt[1]
        zs = pt[2]
        ax.scatter(xs, ys, zs, color=c, facecolor='none',s=100)
ax.view_init(elev=10., azim=100.)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.show()


# Large Solid Circles are Centroids
# 
# Small Circle Facecolor denotes 4th dimension
# 
# Small Circle Outline denotes classification (matching centroid color)

# In[18]:

def findElbow(x):
    error = []
    for i in xrange(10):
        labels, nearests, centers = kmeans(x_iris,i+1)
        error.append(rmsq(nearests,centers))
    plt.figure(figsize=(10,10))
    plt.plot(range(1,11),error,'k',linewidth=10)
    plt.plot(range(1,11),error,'ko',markersize=25)
findElbow(x_iris)


# In[45]:

from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_pickle('articles.pkl')
df.head()


# In[104]:

wordVector = cv(stop_words = 'english', encoding='latin-1')
wordWeights = tfidf(stop_words = 'english', encoding='latin-1')

corpus = df[df['section_name'] == 'Sports']['content']
corpus = corpus.append(df[df['section_name'] == 'Arts']['content'])
corpus = corpus.append(df[df['section_name'] == 'Business Day']['content'])

bag = wordVector.fit_transform(corpus)
weightybags = wordWeights.fit_transform(corpus)


# In[105]:

weightybags = weightybags.toarray()


# In[106]:

bag = bag.toarray()


# In[107]:

features = np.hstack((weightybags,bag))
features.shape


# In[109]:

labels, nearests, centers = kmeans(weightybags,3)


# In[110]:

print labels


# In[132]:

sections = df[df['section_name'] == 'Sports']['section_name']
sections = sections.append(df[df['section_name'] == 'Arts']['section_name'])
sections = sections.append(df[df['section_name'] == 'Business Day']['section_name'])
sections = np.array(sections)
compare = np.hstack((np.array(labels).reshape((725,1)),sections.reshape((725,1))))


# In[123]:

for x in compare:
    print x


# In[137]:

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform


# In[143]:

distxy = squareform(pdist(bag, metric='euclidean'))


# In[146]:

plt.figure(figsize=(15,15))
R = dendrogram(linkage(distxy, method='complete'))

xlabel('points')
ylabel('Height')
suptitle('Cluster Dendrogram', fontweight='bold', fontsize=14);


# In[197]:

ops = df[df['section_name'] == 'Opinion']['content']
opscv = cv(stop_words = 'english', encoding='latin-1')
bags = opscv.fit_transform(ops)
bags = bags.toarray()
vocab = opscv.get_feature_names()

titles = np.array(df[df['section_name'] == 'Opinion']['headline'])


# In[218]:

def getDender(bags, titles, figsize=(40,15)):
    distxy = squareform(pdist(bags, metric='euclidean'))

    plt.figure(figsize=figsize)
    R = dendrogram(linkage(distxy, method='complete'), orientation = 'right', labels=titles)

    xlabel('distance')
    ylabel('headlines')
    suptitle('Cluster Dendrogram', fontweight='bold', fontsize=14)
    


# In[166]:

kt = cluster.KMeans(n_clusters = 4)


# In[169]:

kt.fit_transform(bag)
labels = kt.labels_
centers = kt.cluster_centers_


# In[171]:

print labels
print centers


# In[210]:

#getting top 20 words within clusters of articles 
def topWords(centers, vocab):
    for i, center in enumerate(centers):
        print 'Subpopulation ' + str(i) + ':'
        for j in center.argsort()[-20:][::-1]:
            print '\t' + vocab[j]
        print '\n'


# In[211]:

topWords(centers,vocab)


# In[226]:

wordVector = cv(stop_words = 'english', encoding='latin-1')

corpus = df[df['section_name'] == 'Sports']['content']
corpus = corpus.append(df[df['section_name'] == 'Arts']['content'])
corpus = corpus.append(df[df['section_name'] == 'Business Day']['content'])

head = df[df['section_name'] == 'Sports']['headline']
head = head.append(df[df['section_name'] == 'Arts']['headline'])
head = head.append(df[df['section_name'] == 'Business Day']['headline'])
head = np.array(head)

sec = df[df['section_name'] == 'Sports']['section_name']
sec = sec.append(df[df['section_name'] == 'Arts']['section_name'])
sec = sec.append(df[df['section_name'] == 'Business Day']['section_name'])
sec = np.array(sec)

titles = []
for i in xrange(len(head)):
    titles.append(sec[i] + ' - ' + head[i])
    
bag = wordVector.fit_transform(corpus)
bag = bag.toarray()
vocab = wordVector.get_feature_names()

getDender(bag, titles, figsize=(15,160))
matplotlib.pyplot.savefig('sports_arts_businessday')


# In[223]:

kt.fit_transform(bag)
labels = kt.labels_
centers = kt.cluster_centers_

topWords(centers,vocab)


# In[229]:

getDender(bag.T, vocab, figsize=(15,160))
# this takes too damn long


# In[ ]:



