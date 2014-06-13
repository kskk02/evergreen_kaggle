# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%pylab inline
import numpy as np
import random
from sklearn import cluster, datasets 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# <codecell>

iris = datasets.load_iris()
x_iris = iris.data
y_iris = iris.target

scale = StandardScaler(copy=True)
scale.fit_transform(x_iris);

# <codecell>

def getdistance(x, y):
    return np.sqrt(sum((x-y)**2))

# <codecell>

# randomly sampled guesses

#def initguesses(x,k):
#    # returns list of k guesses
#    return random.sample(x,k)

# <codecell>

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

# <codecell>

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

# <codecell>

def getCentroid(x):
    return 1.0*np.sum(x,axis=0)/x.shape[0]

# <codecell>

centers = initguesses(x_iris, 2) 
len(centers)

# <codecell>

labels,nearest = findnearest(x_iris, centers)
print len(nearest[0])
print len(nearest[1])

# <codecell>

x = np.array([[1,4,-5],[3,-2,9]])
c = getCentroid(x)
print c

# <codecell>

def kmeans(x,n):
    centers = initguesses(x, n)
    centerpair = []
    for i in xrange(15):
        labels,nearests = findnearest(x, centers)
        for j,nearest in enumerate(nearests):
            #print nearest
            c = getCentroid(nearest)
            centers[j] = c
    labels,nearests = findnearest(x, centers)

#     return labels
    return labels, nearests, centers
labels, nearests, centers = kmeans(x_iris,3)
# print labels

# <codecell>

def rmsq(pts,centers):
    sumdist = 0.0
    for i,center in enumerate(centers):
        for pt in pts[i]:
            sumdist += (1.0*getdistance(center,pt))**2
    return sumdist

# <codecell>

error = rmsq(nearests,centers)

# <codecell>

from collections import Counter
cnt = Counter()
for l in labels:
    cnt[l] +=1
print cnt

# <codecell>

k_means = cluster.KMeans(n_clusters=3)
k_means.fit(x_iris) 
print k_means.labels_
# print k_means.cluster_centers_

# <codecell>

cnt = Counter()
for l in y_iris:
    cnt[l] +=1
print cnt

# <codecell>

fig = plt.figure(figsize=(15,15));

ax = fig.add_subplot(111, projection='3d')
#xs = x_iris[:,0]
#ys = x_iris[:,1]
#zs = x_iris[:,2]
#cs = x_iris[:,3]
#ax.scatter(xs, ys, zs, c=cs)

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
    ax.scatter(xs, ys, zs, c=c, s=100)

for label, nearest in enumerate(nearests):
    for pt in nearest:
        c = cmap[label]
        xs = pt[0]
        ys = pt[1]
        zs = pt[2]
        ax.scatter(xs, ys, zs, c=c)
    
plt.show()

# <codecell>

def findElbow(x):
    error = []
    for i in xrange(10):
        labels, nearests, centers = kmeans(x_iris,i+1)
        error.append(rmsq(nearests,centers))
    plt.figure(figsize=(10,10))
    plt.plot(range(1,11),error,'k',linewidth=10)
    plt.plot(range(1,11),error,'ko',markersize=25)
findElbow(x_iris)

# <codecell>


