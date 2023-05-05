# %%
%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sb
from mpl_toolkits.mplot3d import Axes3D


# %%
#scaling, normalization
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
#kmeans, dbscan, hierarchical (sklearn)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
#evaluation
from sklearn.metrics import silhouette_score

#distance matrix (dbscan elbow, hierarchical)
from scipy.spatial.distance import pdist, squareform
# hierarchical (scipy)
from scipy.cluster.hierarchy import linkage, dendrogram

# %%
df = pd.read_csv(r"words_glasgow.csv",skipinitialspace=True)

df2 = df.copy()

df2["perceivability"] = df2[["imageability", "concreteness"]].mean(axis=1)

dfpp1 = df2.drop(["word", "polysemy","length", 'web_corpus_freq'], axis=1)

scaler = MinMaxScaler()

Y=dfpp1.values

Y_minmax = scaler.fit_transform(Y)

scaled_df = pd.DataFrame(data = Y_minmax, columns = dfpp1.columns)

scaled_df.head()

# %%
scaler = MinMaxScaler()

var = ['arousal', 'valence', 'dominance', 'perceivability', 'familiarity', 'aoa','semsize','gender']
'''
for e in var:
    print(e)
    Y = df2[[e]].values
    Y_minmax = scaler.fit_transform(Y)
    plt.hist(Y_minmax, edgecolor='white')
    plt.show()
'''
df2.head()
'''
Y=df2[var].values

scaled_df = pd.DataFrame(data = Y_minmax, columns = df2.columns)

scaled_df.head()

print(Y)
'''

# %%
data_dist = pdist(Y_minmax, metric='euclidean')
data_link = linkage(data_dist, method='ward')

res = dendrogram(data_link, truncate_mode='lastp')

# %%
hier = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward')
hier.fit(Y_minmax)
hier.labels_
hier.n_clusters_
#hier.n_leaves_

plt.scatter( Y[:,0], Y[:,2], s=50, c= hier.labels_, marker='.' )

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()

# %% [markdown]
# ## Impatto emozionale
# 

# %% [markdown]
# 

# %%
df1emo = scaled_df.drop(['concreteness','imageability','familiarity','aoa','semsize','gender','perceivability'], axis=1)

Yemo=df1emo.values


data_dist = pdist(Yemo, metric='euclidean')
data_link = linkage(data_dist, method='ward')


res= dendrogram(data_link, truncate_mode='lastp')


# %%
hier = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
hier.fit(Yemo)
hier.labels_
hier.n_clusters_
#hier.n_leaves_

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Yemo[:,0], Yemo[:,1], Yemo[:,2], s=1,c= hier.labels_, marker='.')


#ax.plot3D(Yemo[:,1], Yemo[:,2], Yemo[:,3], c= hier.labels_, marker='.')

#plt.scatter( Yemo[:,1], Yemo[:,2], s=50, c= hier.labels_, marker='.' )



plt.show()

# %%
hier.labels_

# %%


# %%
plt.scatter( Y[:,0], Y[:,2], s=10, c= hier.labels_, marker='.' )

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()

# %%



sb.pairplot(scaled_df, 
            plot_kws=dict(marker=".", linewidth=1),
            diag_kws=dict(fill=False),
            corner=False
           )


