# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec

#scaling, normalization
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
#kmeans, dbscan, hierarchical (sklearn)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
#evaluation
from sklearn.metrics import silhouette_score
from sklearn import metrics

#distance matrix (dbscan elbow, hierarchical)
from scipy.spatial.distance import pdist, squareform
# hierarchical (scipy)
from scipy.cluster.hierarchy import linkage, dendrogram
#seaborn #bokeh #altair
from sklearn.decomposition import PCA

from yellowbrick.cluster import KElbowVisualizer
from google.colab import files

!pip install pyfim
from fim import apriori

# %%
#caricamento del dataset
df = pd.read_csv('words_glasgow.csv')
#faccio una copia del dataset in caso di manipolazione dati
dfcopy= df.copy()

# %%
#visualizzazione di alcune righe per avere un'idea dei dati
#le prime righe
df.head()

# %%
#le ultime righe
df.tail()

# %%
#visualizzazione della dimensione 
df.shape

# %%
#prima ricognizione indicativa sui dati
df.describe()

# %%
df.describe(include=['O'])

# %%
df.info()

# %% [markdown]
# In this dataset there are 4682 unique words. Each word is examined in regards of different variables. Those variables are lenght, arousal, valence, dominance, concretness, imageability, familiarity, age of aquisition, semsize gender, polysemy and frequence in a google web corpus.`

# %% [markdown]
# testo in corsivo## 1.1 Data Semantics

# %%
colors1 = ['#e5f5f9','#99d8c9','#2ca25f']


# %%
first=df.loc[:,"familiarity":]
aoa_descr=first.describe()

print(aoa_descr.to_latex())

# %% [markdown]
# testo in grassetto
# 
# #### Arousal

# %% [markdown]
# Brief description of the variables

# %%
#NOME VARIABILE DI INTERESSE
refvar="arousal"



#calcolo frequenza assoluta per aoa
df[refvar].value_counts()

#BOXPLOT

plt.rcParams["figure.figsize"] = [16, 8]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa_boxplot = sb.boxplot( x = refvar, data=df, color = colors1[1])

#bellurie
boxplot_str="boxplot"
str1="distribution of"+" "+refvar
str2=boxplot_str+" "+str1
plt.title(str2, size = 18)
aoa_boxplot.set_xlabel(refvar, rotation = "horizontal", size = 16)



#DISTRIBUTION
plt.rcParams["figure.figsize"] = [16, 12]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa1= df[refvar].dropna(0, inplace = False)

data = aoa1
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(data,color = colors1[1], bins=range(1,10),edgecolor='black')

# Set the ticks to be at the edges of the bins.
ax.set_xticks(bins)

# Label the raw counts and the percentages below the x-axis...
#bellurie
plt.title(str1, size = 18)
plt.xlabel(xlabel = refvar, fontsize = 20, labelpad=20)
plt.ylabel(ylabel = "Occurences", fontsize = 16)
sb.despine(right = True)
plt.show()

aoa_descr=df[refvar].describe()

print(aoa_descr.to_latex())

# %% [markdown]
# Other comments

# %% [markdown]
# 
# 
# #### Valence

# %% [markdown]
# Brief description of the variables

# %%
#NOME VARIABILE DI INTERESSE
refvar="valence"



#calcolo frequenza assoluta per aoa
df[refvar].value_counts()

#BOXPLOT

plt.rcParams["figure.figsize"] = [16, 8]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa_boxplot = sb.boxplot( x = refvar, data=df, color = colors1[1])

#bellurie
boxplot_str="boxplot"
str1="distribution of"+" "+refvar
str2=boxplot_str+" "+str1
plt.title(str2, size = 18)
aoa_boxplot.set_xlabel(refvar, rotation = "horizontal", size = 16)



#DISTRIBUTION
plt.rcParams["figure.figsize"] = [16, 12]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa1= df[refvar].dropna(0, inplace = False)

data = aoa1
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(data,color = colors1[1], bins=range(1,10),edgecolor='black')

# Set the ticks to be at the edges of the bins.
ax.set_xticks(bins)

# Label the raw counts and the percentages below the x-axis...
#bellurie
plt.title(str1, size = 18)
plt.xlabel(xlabel = refvar, fontsize = 20, labelpad=20)
plt.ylabel(ylabel = "Occurences", fontsize = 16)
sb.despine(right = True)
plt.show()

aoa_descr=df[refvar].describe()

print(aoa_descr.to_latex())

# %%


# %% [markdown]
# Other comments

# %%


# %% [markdown]
# 
# 
# #### Dominance

# %% [markdown]
# Brief description of the variables

# %%
refvar="dominance"

# %%
#calcolo frequenza assoluta per aoa
df[refvar].value_counts()

#BOXPLOT

plt.rcParams["figure.figsize"] = [16, 8]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa_boxplot = sb.boxplot( x = refvar, data=df, color = colors1[1])

#bellurie
boxplot_str="boxplot"
str1="distribution of"+" "+refvar
str2=boxplot_str+" "+str1
plt.title(str2, size = 18)
aoa_boxplot.set_xlabel(refvar, rotation = "horizontal", size = 16)



#DISTRIBUTION
plt.rcParams["figure.figsize"] = [16, 12]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa1= df[refvar].dropna(0, inplace = False)

data = aoa1
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(data,color = colors1[1], bins=range(1,10),edgecolor='black')

# Set the ticks to be at the edges of the bins.
ax.set_xticks(bins)

# Label the raw counts and the percentages below the x-axis...
#bellurie
plt.title(str1, size = 18)
plt.xlabel(xlabel = refvar, fontsize = 20, labelpad=20)
plt.ylabel(ylabel = "Occurences", fontsize = 16)
sb.despine(right = True)
plt.show()

aoa_descr=df[refvar].describe()

print(aoa_descr.to_latex())

# %% [markdown]
# Other comments

# %% [markdown]
# 
# 
# #### Concreteness

# %% [markdown]
# (numerical): represents the degree to which something can be experienced by our senses. It ranges from 1, abstract words, to 7, concrete words. 

# %%
#NOME VARIABILE DI INTERESSE
refvar="concreteness"



#calcolo frequenza assoluta per aoa
df[refvar].value_counts()

#BOXPLOT

plt.rcParams["figure.figsize"] = [16, 8]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa_boxplot = sb.boxplot( x = refvar, data=df, color = colors1[1])

#bellurie
boxplot_str="boxplot"
str1="distribution of"+" "+refvar
str2=boxplot_str+" "+str1
plt.title(str2, size = 18)
aoa_boxplot.set_xlabel(refvar, rotation = "horizontal", size = 16)



#DISTRIBUTION
plt.rcParams["figure.figsize"] = [16, 12]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa1= df[refvar].dropna(0, inplace = False)

data = aoa1
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(data,color = colors1[1], bins=range(1,8),edgecolor='black')

# Set the ticks to be at the edges of the bins.
ax.set_xticks(bins)

# Label the raw counts and the percentages below the x-axis...
#bellurie
plt.title(str1, size = 18)
plt.xlabel(xlabel = refvar, fontsize = 20, labelpad=20)
plt.ylabel(ylabel = "Occurences", fontsize = 16)
sb.despine(right = True)
plt.show()

aoa_descr=df[refvar].describe()

print(aoa_descr.to_latex())

# %% [markdown]
# Other comments

# %%


# %% [markdown]
# 
# 
# #### Imageability

# %% [markdown]
# (numerical): measures how difficult is to generate a mental image of something. It ranges from 1, hard to imagine, to 7, easy to imagine. 

# %%
#NOME VARIABILE DI INTERESSE
refvar="imageability"



#calcolo frequenza assoluta per aoa
df[refvar].value_counts()

#BOXPLOT

plt.rcParams["figure.figsize"] = [16, 8]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa_boxplot = sb.boxplot( x = refvar, data=df, color = colors1[1])

#bellurie
boxplot_str="boxplot"
str1="distribution of"+" "+refvar
str2=boxplot_str+" "+str1
plt.title(str2, size = 18)
aoa_boxplot.set_xlabel(refvar, rotation = "horizontal", size = 16)



#DISTRIBUTION
plt.rcParams["figure.figsize"] = [16, 12]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa1= df[refvar].dropna(0, inplace = False)

data = aoa1
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(data,color = colors1[1], bins=range(1,8),edgecolor='black')

# Set the ticks to be at the edges of the bins.
ax.set_xticks(bins)

# Label the raw counts and the percentages below the x-axis...
#bellurie
plt.title(str1, size = 18)
plt.xlabel(xlabel = refvar, fontsize = 20, labelpad=20)
plt.ylabel(ylabel = "Occurences", fontsize = 16)
sb.despine(right = True)
plt.show()

aoa_descr=df[refvar].describe()

print(aoa_descr.to_latex())

# %% [markdown]
# Other comments

# %%


# %% [markdown]
# 
# 
# #### Familiarity

# %% [markdown]
# (numerical): is a measure of a word’s subjective experience. The range is from 1 (unfamiliar) to 7 (familiar). 
# 

# %%
#NOME VARIABILE DI INTERESSE
refvar="familiarity"



#calcolo frequenza assoluta per aoa
df[refvar].value_counts()

#BOXPLOT

plt.rcParams["figure.figsize"] = [16, 8]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa_boxplot = sb.boxplot( x = refvar, data=df, color = colors1[1])

#bellurie
boxplot_str="boxplot"
str1="distribution of"+" "+refvar
str2=boxplot_str+" "+str1
plt.title(str2, size = 18)
aoa_boxplot.set_xlabel(refvar, rotation = "horizontal", size = 16)



#DISTRIBUTION
plt.rcParams["figure.figsize"] = [16, 12]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa1= df[refvar].dropna(0, inplace = False)

data = aoa1
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(data,color = colors1[1], bins=range(1,8),edgecolor='black')

# Set the ticks to be at the edges of the bins.
ax.set_xticks(bins)

# Label the raw counts and the percentages below the x-axis...
#bellurie
plt.title(str1, size = 18)
plt.xlabel(xlabel = refvar, fontsize = 20, labelpad=20)
plt.ylabel(ylabel = "Occurences", fontsize = 16)
sb.despine(right = True)
plt.show()

aoa_descr=df[refvar].describe()

print(aoa_descr.to_latex())

# %% [markdown]
# Other comments

# %%


# %% [markdown]
# #### Age of Aquisition

# %% [markdown]
# The variable age of aquisition (aoa) indicates the supposed age in which that person first learned that specific word. Clearly it is not easy to remember at which age a word is learned, and for that reason in this dataset this variables refears to an estimation of the age of aquisition. The scale is defined as a series of consecutive 2-year periods from the ages of 2 and 12 years, and a final period referring to 13 years and older. This shows how we have 7 different ranges, 0-2, 2-4, 4-6, 6-8, 8-10-, 10-12 and 13+.

# %%
#calcolo frequenza assoluta per aoa
df["aoa"].value_counts()

# %%
#imposto lo spazio
colors1 = ['#e5f5f9','#99d8c9','#2ca25f']
plt.rcParams["figure.figsize"] = [16, 4]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})
plt.subplot(1, 2, 1)
#distribuzione e tendenze centrali: boxplot
aoa_boxplot = sb.boxplot( x = "aoa", data=df, color = colors1[1])
plt.title("Boxplot distribution of age of aquisition", size = 18)
aoa_boxplot.set_xlabel("Age of aquisition", rotation = "horizontal", size = 16)
plt.subplot(1, 2, 2)

aoa1= df["aoa"].dropna(0, inplace = False)

#imposto la legenda (da rivedere)
#lables_aoa= [0-2, 2-4, 4-6, 6-8, 8-10, 10-12, 13]
#plt.legend(title="Age of aquisition ranges", title_fontsize = 14, bbox_to_anchor=(1, 1), labels=lables_aoa )

#distribuzione e tendenze centrali: istogramma
aoa_hist = plt.hist(aoa1,  color = colors1[1],edgecolor='black',bins=range(0,8))

plt.title("Distribution of aoa", size = 18)
plt.xlabel(xlabel = "Age of Aquisition", fontsize = 16)
plt.ylabel(ylabel = "Occurences", fontsize = 16)
sb.despine(right = True)

plt.show()

# %%
colors1 = ['#e5f5f9','#99d8c9','#2ca25f']
plt.rcParams["figure.figsize"] = [16, 8]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa_boxplot = sb.boxplot( x = "aoa", data=df, color = colors1[1])
plt.title("Boxplot distribution of age of aquisition", size = 18)
aoa_boxplot.set_xlabel("Age of aquisition", rotation = "horizontal", size = 16)

plt.show()

# %%
colors1 = ['#e5f5f9','#99d8c9','#2ca25f']


#BOXPLOT

plt.rcParams["figure.figsize"] = [16, 8]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa_boxplot = sb.boxplot( x = "aoa", data=df, color = colors1[1])
plt.title("Boxplot distribution of Age of Aquisition", size = 18)
bin_centers = 0.5 * np.diff(bins) + bins[:-1]
for i, x in zip(range(0,7), bin_centers):
    # Label the raw counts
    age_range=['0-2','2-4','4-6','6-8','8-10','10-12','13+']
    ax.annotate(age_range[i], xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -22), textcoords='offset points',fontsize=16, va='top', ha='center')


aoa_boxplot.set_xlabel("Age of Aquisition", rotation = "horizontal", size = 16)



#DISTRIBUTION
plt.rcParams["figure.figsize"] = [16, 8]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa1= df["aoa"].dropna(0, inplace = False)

data = aoa1
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(data,color = colors1[1], bins=range(0,8),edgecolor='black')

# Set the ticks to be at the edges of the bins.
ax.set_xticks(bins)

# Label the raw counts and the percentages below the x-axis...
bin_centers = 0.5 * np.diff(bins) + bins[:-1]
for i, x in zip(range(0,7), bin_centers):
    # Label the raw counts
    age_range=['0-2','2-4','4-6','6-8','8-10','10-12','13+']
    ax.annotate(age_range[i], xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -22), textcoords='offset points',fontsize=16, va='top', ha='center')

    # Label the percentages
    #percent = '%0.0f%%' % (100 * float(count) / counts.sum())
    #ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
        #xytext=(0, -32), textcoords='offset points', va='top', ha='center')


# Give ourselves some more room at the bottom of the plot
#plt.subplots_adjust(bottom=0.50)
plt.title("Distribution of aoa", size = 18)
plt.xlabel(xlabel = "Age of Aquisition", fontsize = 20, labelpad=20)
plt.ylabel(ylabel = "Occurences", fontsize = 16)
sb.despine(right = True)
plt.show()

aoa_descr=df["aoa"].describe()

print(aoa_descr.to_latex())

# %%
colors1 = ['#e5f5f9','#99d8c9','#2ca25f']

refvar="aoa"

#BOXPLOT

plt.rcParams["figure.figsize"] = [16, 8]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa_boxplot = sb.boxplot( x = refvar, data=df, color = colors1[1])

#bellurie

plt.title("Boxplot distribution of aoa", size = 18)
aoa_boxplot.set_xlabel("Age of Aquisition", rotation = "horizontal", size = 16)



#DISTRIBUTION
plt.rcParams["figure.figsize"] = [16, 8]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa1= df[refvar].dropna(0, inplace = False)

data = aoa1
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(data,color = colors1[1], bins=range(1,8),edgecolor='black')

# Set the ticks to be at the edges of the bins.
ax.set_xticks(bins)

# Label the raw counts and the percentages below the x-axis...
bin_centers = 0.5 * np.diff(bins) + bins[:-1]
for i, x in zip(range(0,7), bin_centers):
    # Label the raw counts
    age_range=['0-2','2-4','4-6','6-8','8-10','10-12','13+']
    ax.annotate(age_range[i], xy=(x, 0), xycoords=('data', 'axes fraction'),
        xytext=(0, -22), textcoords='offset points',fontsize=16, va='top', ha='center')

    # Label the percentages
    #percent = '%0.0f%%' % (100 * float(count) / counts.sum())
    #ax.annotate(percent, xy=(x, 0), xycoords=('data', 'axes fraction'),
        #xytext=(0, -32), textcoords='offset points', va='top', ha='center')


# Give ourselves some more room at the bottom of the plot
plt.subplots_adjust(bottom=0.50)

# Label the raw counts and the percentages below the x-axis...
#bellurie
plt.title("Distribution of aoa", size = 18)
plt.xlabel(xlabel = "Age of Aquisition", fontsize = 20, labelpad=20)
plt.ylabel(ylabel = "Occurences", fontsize = 16)
sb.despine(right = True)
plt.show()

aoa_descr=df[refvar].describe()

print(aoa_descr.to_latex())

# %%


# %%
aoa_descr=df["aoa"].describe()

print(aoa_descr.to_latex())

# %% [markdown]
# There are 4682 occurrences for this variable. That means that there are no missing values, or NaN. The mean is 4.14, and the standard deviation is 1.25.

# %% [markdown]
# #### Size

# %% [markdown]
# In this dataset "size" is a measure of magnitude expressed in either concrete or abstarct terms (big, small). That is, if a word can be associated with adjectives like big or small (e.g. palace or mountain for concrete object, and knowledge or love for abstarct ideas).

# %%
#calcolo frequenza assoluta per aoa
df["semsize"].value_counts()

# %%
#imposto lo spazio
colors1 = ['#e5f5f9','#99d8c9','#2ca25f']
plt.rcParams["figure.figsize"] = [16, 4]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})
plt.subplot(1, 2, 1)
#distribuzione e tendenze centrali: boxplot
aoa_boxplot = sb.boxplot( x = "semsize", data=df, color = colors1[1])
plt.title("Boxplot distribution of semantic size", size = 18)
aoa_boxplot.set_xlabel("semantic size", rotation = "horizontal", size = 16)
plt.subplot(1, 2, 2)

size1= df["semsize"].dropna(0, inplace = False)

#imposto la legenda (da rivedere)
#lables_size= [0-2, 2-4, 4-6, 6-8, 8-10, 10-12, 13]
#plt.legend(title="Age of aquisition ranges", title_fontsize = 14, bbox_to_anchor=(1, 1), labels=lables_size )
#distribuzione e tendenze centrali: istogramma
aoa_hist = plt.hist(size1,  color = colors1[1],edgecolor='black',bins=range(1,8))

#BELLURIE

plt.title("Distribution of semantic size", size = 18)
plt.xlabel(xlabel = "semantic size", fontsize = 16)
plt.ylabel(ylabel = "Degree", fontsize = 16)
sb.despine(right = True)

plt.show()

# %%


# %%


# %%
colors1 = ['#e5f5f9','#99d8c9','#2ca25f']

refvar="semsize"

#BOXPLOT

plt.rcParams["figure.figsize"] = [16, 8]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa_boxplot = sb.boxplot( x = refvar, data=df, color = colors1[1])

#bellurie

plt.title("Boxplot distribution of semantic size", size = 18)
aoa_boxplot.set_xlabel("Semantic Size", rotation = "horizontal", size = 16)



#DISTRIBUTION
plt.rcParams["figure.figsize"] = [16, 12]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa1= df[refvar].dropna(0, inplace = False)

data = aoa1
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(data,color = colors1[1], bins=range(1,8),edgecolor='black')

# Set the ticks to be at the edges of the bins.
ax.set_xticks(bins)

# Label the raw counts and the percentages below the x-axis...
#bellurie
plt.title("Distribution of semsize", size = 18)
plt.xlabel(xlabel = "Semantic Size", fontsize = 20, labelpad=20)
plt.ylabel(ylabel = "Occurences", fontsize = 16)
sb.despine(right = True)
plt.show()

aoa_descr=df[refvar].describe()

print(aoa_descr.to_latex())

# %%
df["semsize"].describe()

# %% [markdown]
# The count of occurrences for this variable suggests that there are no missing values. The mean is 4.13 and the standard deviation is just above 1. This results suggests that the tendency is that is more common to think that a word has a bigger semantic size.

# %% [markdown]
# #### Gender

# %% [markdown]
# Gender in this dataset refers to how strongly its meaning is associated with male or female behaviour or idea. This variable could be very interesting in regards of the social bias that might, or might not, be present.

# %%
#calcolo frequenza assoluta per aoa
df["gender"].value_counts()

# %%
#imposto lo spazio
colors1 = ['#e5f5f9','#99d8c9','#2ca25f']
plt.rcParams["figure.figsize"] = [16, 4]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})
plt.subplot(1, 2, 1)
#distribuzione e tendenze centrali: boxplot
gender_boxplot = sb.boxplot( x = "gender", data=df, color = colors1[1])
plt.title("Boxplot distribution of perceived gender", size = 18)
gender_boxplot.set_xlabel("gender", rotation = "horizontal", size = 16)
plt.subplot(1, 2, 2)

gender1= df["gender"].dropna(0, inplace = False)

#imposto la legenda
lables_size= [0-2, 2-4, 4-6, 6-8, 8-10, 10-12, 13]
plt.legend(title="Perceived gender", title_fontsize = 14, bbox_to_anchor=(1, 1), labels=lables_size )
#distribuzione e tendenze centrali: istogramma
aoa_hist = plt.hist(gender1,  color = colors1[1])
plt.title("Distribution of perceived gender", size = 18)
plt.xlabel(xlabel = "gender", fontsize = 16)
plt.ylabel(ylabel = "Degree", fontsize = 16)
sb.despine(right = True)

plt.show()

# %%
df["gender"].describe()

# %%
df["length"].describe()

# %%
import matplotlib.ticker as mticker
colors1 = ['#e5f5f9','#99d8c9','#2ca25f']

refvar="gender"

#BOXPLOT

plt.rcParams["figure.figsize"] = [16, 8]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa_boxplot = sb.boxplot( x = refvar, data=df, color = colors1[1])

#bellurie

plt.title("Boxplot distribution of gender", size = 18)
aoa_boxplot.set_xlabel("Gender", rotation = "horizontal", size = 16)



#DISTRIBUTION
plt.rcParams["figure.figsize"] = [16, 12]
sb.set_context("notebook", font_scale=1.5, rc={"font.size":14,"axes.titlesize": 16,"axes.labelsize": 12})

aoa1= df[refvar].dropna(0, inplace = False)

data = aoa1
fig, ax = plt.subplots()
counts, bins, patches = ax.hist(data,color = colors1[1], bins=range(1,8),edgecolor='black')

# Set the ticks to be at the edges of the bins.
ax.set_xticks(bins)

# Label the raw counts and the percentages below the x-axis...
#bellurie
plt.title("Distribution of gender", size = 18)
plt.xlabel(xlabel = "Gender", fontsize = 20, labelpad=20)


plt.ylabel(ylabel = "Occurences", fontsize = 16)


ax.xaxis.set_minor_locator(mticker.FixedLocator((1, 7)))
ax.xaxis.set_minor_formatter(mticker.FixedFormatter(("Label A", "Label B")))
plt.setp(ax.yaxis.get_minorticklabels(), size=15, va="center")
ax.tick_params("x",which="minor",pad=25, left=False)

sb.despine(right = True)
plt.show()

aoa_descr=df[refvar].describe()

print(aoa_descr.to_latex())

# %% [markdown]
# This variable is not self intuitive. There is no visible correlation from the number and the perceived gender of that word. A supposition is that the higher the value, the more "masculine" the word is perceived. For example the word "actor" has a perceived gender value of 5.588, where the max value is 6.971.
# 
# Also it seems that the distribution is heavly centered, leaving a significant amount of outliers, as seen in the figure above.

# %%
 df[["word","gender"]].loc[[51, 52]]

# %% [markdown]
# ## 1.2    Distribution of variables and statistics
# 

# %% [markdown]
# Codice per le distribuzioni delle variabili 'arousal', 'valence', 'dominance', 'concreteness','imageability', 'familiarity'
# 

# %%
df.corr() #matrice di correlazione

#distribuzione delle variabili
scaler = MinMaxScaler()

var = ['arousal', 'valence', 'dominance', 'concreteness',
       'imageability', 'familiarity']

for e in var:
    print(e)
    Y = df[[e]].values
    Y_minmax = scaler.fit_transform(Y)
    plt.hist(Y_minmax, edgecolor='white')
    plt.show()

#piechart polisemia

#percentuale delle polisemiche 
m = 0
p = 0

for e in df['polysemy'].values:
    if e == 1:
        p += 1
    if e == 0:
        m += 1

p1 = p/(p+m)*100
m1 = m/(p+m)*100
print(p1, m1)

#grafico
labels = 'Polisemic', 'Monosemic'
sizes = [p1, m1]
explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

#matrice di correlazione
corr=df.corr()
plt.figure(figsize=(16, 6))
heatmap = sb.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
#sb.heatmap(corr, cmap="Blues", annot=True)



# %% [markdown]
# ### 1.2.1    Variable Normalization

# %% [markdown]
# ## 1.3    Assessing Data Quality

# %% [markdown]
# ### 1.3.1    Missing Values

# %%
df.isnull().sum()

# %% [markdown]
# The dataset seems to be almost without null values. In fact, there are only 14 NaN, and all are concentrated in the "web_corpus_freq" variable.

# %%
df[df['web_corpus_freq'].isnull()]

# %% [markdown]
# Looking at the values that are present in the dataset but are not present in the web corpus, we can spot some similarities.
# There are 3 recurrences of the same word but slightly different from one another: "Mom", "Mum" and "Mommy". Also the word "Dad" is not present in the corpus.
# 
# In addition, there are 3 words written with full capital letters: "FALSE", "TRUE", and "TV". If the corpus is case sensitive, that could be an explaination of why those words are not present.
# 
# Finally, there are 2 words of social media that are absent in the corpus: "Facebook" and "Twitter". This is a bit strange since those 2 words are really known.

# %% [markdown]
# ### 1.3.2    Outliers

# %%
#creating a copy of the df
df2 = df.copy()
#dropping some variables for boxplot visualzation
df3 = df2.drop(["word","web_corpus_freq", "polysemy","length"], axis=1)
df3.head()

# %%
#displaying all boxplot in one plot

#defining outliers as a blue underscore circle
blue_circle = dict(markerfacecolor='blue', marker='o', markeredgecolor='white')

#defining fig, axs as plt.subplots. In the arguments we pass in that we want 1 row, and then the number of columns. 
#The number of columns will be equal to the length of df.columns. 
#Finally, we will set the figsize to 20 by 10.
fig, axs = plt.subplots(1, len(df3.columns), figsize=(20,10))

#defining the shape and color of the mean
mean_shape = dict(markerfacecolor='green', marker='D', markeredgecolor='green')

#creating a small for loop which will loop over each of our plot axes. 
#Using the enumerate function here to keep track of i, our index value
for i, ax in enumerate(axs.flat):
    #addong in our boxplot by calling upon ax.boxplot() and passing in df.iloc, 
    #which allows us to get columns by the index variable, i. 
    #We do this by creating square brackets, with a colon followed by a comma. 
    #This tells the iloc function to get all rows, and then we pass in the column index. 
    #Next we specify our outlier properties and pass in red_circle.
    ax.boxplot(df3.iloc[:,i], flierprops=blue_circle, showmeans = True, meanprops =mean_shape, notch=True)
    ax.set_title(df3.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    
plt.tight_layout()

# %% [markdown]
# First, some variables had to be dropped to perform this analysys with boxplots. "word" is not a numerical variable, and for that reason is not used in this plot. "web_corpus-freq" and "polysemy" can be analised on their own.

# %% [markdown]
# The plot shows that some variables are well distributed, like "concretness" and "imageability", while others are not, like "length", "arousal" and "dominance". 
# Later this issue will be addressed with more depth.

# %%
df["polysemy"].plot(kind="box", vert = False)

# %% [markdown]
# "polysemy" has only 2 possible values, 0 and 1, that are false and true. For that reason it doesn't have outliers.

# %%
df["polysemy"].value_counts().plot(kind="pie")

# %% [markdown]
# "web_corpus_freq" instead seems to have a lot of outliers, as shown below.

# %%
df["web_corpus_freq"].dropna(0, inplace = False).plot(kind="box", vert = False, logx=True)

# %% [markdown]
# ### 1.3.3    Errors

# %%


# %%
var = ['arousal', 'valence', 'dominance', 'concreteness',
       'imageability', 'familiarity','semsize','gender','aoa']

for element in var:
  for e in df[element].values:
    if(type(e)!=np.float64):
       print(type(e))

  print("tutto ok fra")
    

# %%
#creating a copy of the df
df_errors = df.copy()

# %%
df_errors[["word", "length"]]

# %%
df_errors["word"] = df_errors.word.str.len()

# %%
df_errors[["word", "length"]]

# %%
df_errors["word"].equals(df_errors["length"])

# %% [markdown]
# All values of the variable 'word' are correct in regards of their length

# %% [markdown]
# ### 1.3.4    Semantic Inconsistencies

# %%
df.loc[df['polysemy']==1]

# %% [markdown]
# ## 1.4    Eventual other variable trasformations

# %% [markdown]
# ## 1.5    Pairwise correlations and eventual elimination of variables

# %% [markdown]
# An overview of the relation between the nine variables is provided in Fig. 6. Where a correlation greater than |0.6| is found, we plotted the values of the two variables for a better visualization (fig. 7). 
# There is a strong correlation (0.91) between concreteness and imageability: it is difficult to imagine an abstract word and easier to imagine a concrete one. Moreover, concreteness and imageability relate to the other variables similarly, with a margin of +-0.14. Therefore we merged them into a new variable, perceivability. The values of perceivabilty are the mean of concreteness and imageability values. 
# Other positively correlated variables are valence and dominance, with 0.72: the more valuable an item is perceived, the higher the degree of control over the object. 
# Familiary and age of acquisition are instead negatively related: from the pairplot (Fig \ref{fig:pairplot}) is apparent that every word acquired in early age is highly familiar. 

# %%
#matrice di correlazione
corr=df.corr()
plt.figure(figsize=(16, 6))
heatmap = sb.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
#sb.heatmap(corr, cmap="Blues", annot=True)


# %%
#creating a copy of the df
dfpp = df.copy()
#dropping some variables for boxplot visualzation
dfpp1 = dfpp.drop(["word", "polysemy","length", 'gender', 'web_corpus_freq', 'semsize', 'arousal'], axis=1)

# %%
sb.pairplot(dfpp1, 
            plot_kws=dict(marker=".", linewidth=1),
            diag_kws=dict(fill=False),
            corner=False
           )

# %%
sb.pairplot(dfpp1, 
            x_vars=['concreteness'],
            y_vars=['imageability'],
            plot_kws=dict(marker=".", linewidth=1),
            diag_kws=dict(fill=False),
            corner=False
           )

#plt.savefig('pp_imageability_concreteness.png', dpi=300)
#files.download('pp_imageability_concreteness.png') 

# %%
pp_dv = sb.pairplot(dfpp1, 
            x_vars=['valence'],
            y_vars=['dominance'],
            plot_kws=dict(marker=".", linewidth=1),
            diag_kws=dict(fill=False),
            corner=False
           )

#plt.savefig('pp_dominance_valence.png', dpi=300)
#files.download('pp_dominance_valence.png') 

# %%
sb.pairplot(dfpp1, 
            x_vars=['familiarity'],
            y_vars=['aoa'],
            plot_kws=dict(marker=".", linewidth=1),
            diag_kws=dict(fill=False),
            corner=False
           )

#plt.savefig('pp_aoa_familiarity.png', dpi=300)
#files.download('pp_aoa_familiarity.png') 

# %%
df2["perceivability"] = df2[["imageability", "concreteness"]].mean(axis=1)
df2.head()

# %%
df_perc=df2.drop(["concreteness","imageability"], axis=1)
df_perc.head()

# %% [markdown]
# # 2    Clustering

# %% [markdown]
# ## 2.1    Preprocessing
# 
# 
# 

# %%
dfprepro= df_perc.copy()
dfprepro=dfprepro.rename(columns={"gender": "masculinity"})

# %%
dfprepro.loc[(dfprepro['web_corpus_freq'].isnull() == True), 'web_corpus_freq'] = dfprepro['web_corpus_freq'].mean()


# %%
dfprepro[dfprepro['web_corpus_freq'].isnull()]

# %%
dfprepro["web_corpus_log"] = pd.qcut(dfprepro["web_corpus_freq"], 10) #taglio la variabile web_corpus_freq in 10 gruppi
print(dfprepro[["web_corpus_log", "web_corpus_freq"]].groupby(["web_corpus_log"], as_index=False).mean())

# %%
dataframe = [dfprepro]

# %%
for dataset in dataframe:
    dataset.loc[(dataset["web_corpus_freq"] > 10000) & (dataset["web_corpus_freq"] <= 100000), "web_corpus_freq"] = 4
    dataset.loc[(dataset["web_corpus_freq"] > 100000) & (dataset["web_corpus_freq"] <= 1000000), "web_corpus_freq"] = 5
    dataset.loc[(dataset["web_corpus_freq"] > 1000000) & (dataset["web_corpus_freq"] <= 10000000), "web_corpus_freq"] = 6
    dataset.loc[(dataset["web_corpus_freq"] > 10000000) & (dataset["web_corpus_freq"] <= 100000000), "web_corpus_freq"] = 7
    dataset.loc[(dataset["web_corpus_freq"] > 100000000) & (dataset["web_corpus_freq"] <= 1000000000), "web_corpus_freq"] = 8
    dataset.loc[dataset["web_corpus_freq"] > 1000000000, "web_corpus_freq"] = 9
    
dfprepro.head()

# %%
dfprepro["web_corpus_freq"].plot(kind="hist")

# %%
dfprepro['web_corpus_freq'].describe()

# %%
dfprepro['web_corpus_log'].describe()

# %%
dfprepro = dfprepro.drop(["web_corpus_log","word"], axis=1)
#dfprepro.loc[(dfprepro['web_corpus_freq'].isnull() == True), 'web_corpus_freq'] = dfprepro['web_corpus_freq'].mean()
dfprepro.isnull().sum()
dfprepro.head()

# %%
pca = PCA(n_components=2)
pca.fit(dfprepro)
PCA_df = pd.DataFrame(pca.transform(dfprepro))
pca.transform(dfprepro)
score_pca = pca.transform(dfprepro)

# %%


# %%
scaler = MinMaxScaler()

Y_pca=PCA_df.values

Y_pca_minmax = scaler.fit_transform(Y_pca)

scaled_dfprepro = pd.DataFrame(data = Y_pca_minmax, columns = PCA_df.columns)

scaled_dfprepro.head()

X=dfprepro.values

X_minmax = scaler.fit_transform(X)

df_xminmax = pd.DataFrame(data = X_minmax, columns = dfprepro.columns)



# %%
scaled_dfprepro.head()

# %%
df_xminmax['word']=df_perc['word']

# %%
cols = df_xminmax.columns.tolist()
cols = cols[-1:] + cols[:-1]

df_xminmax=df_xminmax[cols]


df_xminmax.head()

# %%
df_xminmax=df_xminmax.drop(['word'],axis=1)

df_xminmax.describe()

# %%
PCA_df

# %%
x = PCA_df[0]
y = PCA_df[1]
#z = PCA_df[2]
print(type (x))
print(type (y))
#print(type (z))
print(type (Y_pca_minmax))

# %%
print (Y_pca_minmax)

# %% [markdown]
# ## 2.2    Clustering analysis by K-Means

# %%
kmeans = KMeans( n_clusters=3, n_init=10, max_iter=100 )
kmeans.fit(Y_pca_minmax)

# %%
kmeans.labels_

# %%
print('SSE', kmeans.inertia_)
print('Silhouette', silhouette_score(Y_pca_minmax, kmeans.labels_))

# %%
print('Calinski-Harabasz', metrics.calinski_harabasz_score(Y_pca_minmax, kmeans.labels_))

# %%
silhouette_list = []

#for k in range(2, 50):
#    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=100)
#    kmeans.fit(Y_pca_minmax)
#    silhouette_list.append( silhouette_score(Y_pca_minmax, kmeans.labels_))

# %%
#plt.plot(range(2, len(silhouette_list)+2), silhouette_list, marker='*')
#plt.ylabel('Silhouette score')
#plt.show()

# %%
kmeans = KMeans(n_clusters=3, n_init=10, max_iter=100)
kmeans.fit(Y_pca_minmax)

# %%
centers = kmeans.cluster_centers_

# %%
kmeans.labels_

# %%
from matplotlib import cm
plt.scatter( Y_pca[:,0], Y_pca[:,1], s=40, c= kmeans.labels_ , cmap = cm.tab20c)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()


# %%
plt.scatter( Y_pca_minmax[:,0], Y_pca_minmax[:,1], s=40, c= kmeans.labels_ , cmap = cm.tab20c)
plt.scatter( centers[:,0], centers[:,1], c='red', marker='*', s=200 )

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()

# %%
scaled_dfprepro['cluster'] = kmeans.labels_

df_xminmax['cluster'] = kmeans.labels_

df_xminmax['cluster']=df_xminmax['cluster'].map({0:'1', 1:'2', 2:'3'})

df_xminmax_plot=df_xminmax.drop(["polysemy"], axis=1)

plt.figure(figsize = (8,8))
pie = scaled_dfprepro.groupby(['cluster']).size().to_frame().reset_index()
pie.rename(columns={0: 'count'}, inplace=True)

pie_labels = ['first cluster', 'second cluster', 'third cluster']
plt.pie(pie['count'], labels=pie_labels)
plt.show()
sb.set(font_scale=3.7)

fig, axs = plt.subplots(ncols=2, nrows=6, figsize=(30,60))
for cols, x in zip(df_xminmax_plot.columns[:5], range(5)):
    for y in range(1):
        g1=sb.boxplot(y=cols, x='cluster', data=df_xminmax_plot, order=['1', '2','3'], ax=axs[x,y], showfliers = True )
        g1.set(xlabel=None)

for cols, x in zip(df_xminmax_plot.columns[5:], range(5)):
    for y in range(1,2):
        g2=sb.boxplot(y=cols, x='cluster', data=df_xminmax_plot, order=['1', '2','3'], ax=axs[x,y], showfliers = True )
        g2.set(xlabel=None)

# %%
df_perc.isnull().sum()

df_xminmax_plot['word']=df_perc['word']
df_xminmax_plot.loc[df_xminmax['cluster']=='First']

# %%
df_xminmax_plot.loc[df_xminmax['cluster']=='Second']

# %%
df_xminmax_plot.loc[df_xminmax['cluster']=='Third']

# %% [markdown]
# ## 2.3    Analysis by density-based clustering

# %%
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=20)
neighbors_fit = neighbors.fit(Y_pca_minmax)
distances, indices = neighbors_fit.kneighbors(Y_pca_minmax)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)

# %%
fig = plt.figure(figsize=(20, 12))
fig.subplots_adjust(hspace=.5, wspace=.2)
i = 1

for x in range(0, 15, 1):
    #y=-2+x/10
    eps = 0.012+1*x/1000
    db = DBSCAN(eps=eps, min_samples=20).fit(Y_pca_minmax)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    #print(eps)
    ax = fig.add_subplot(3, 5, i)
    ax.title.set_text("eps = {}".format(round(eps, 3)))
    #ax.text(1, 4, "eps = {}".format(round(eps, 3)), fontsize=25, ha="center")
    sb.scatterplot(Y_pca_minmax[:,0], Y_pca_minmax[:,1], hue=["{} cluster".format(x) for x in labels],legend=False,palette='tab20c',markers='.',size=1)
    ax.text(0.5,-0.3, "{} cluster(s)".format(max(labels)+1), size=12, ha="center", transform=ax.transAxes)


    i += 1

# %%
dbscan = DBSCAN( eps=0.018, min_samples=20)
dbscan.fit(Y_pca_minmax)

# %%
dbscan.labels_

# %%
unique_labels, unique_counts = np.unique(dbscan.labels_, return_counts=True)
print(max(unique_labels))

# %%
#cols = [col for col in PCA_df.columns if col != 'class']

plt.scatter( Y_pca_minmax[:,0], Y_pca_minmax[:,1], s=25, c= dbscan.labels_, cmap = cm.tab20 )

plt.xticks(fontsize=2)
plt.yticks(fontsize=2)

plt.show()

# %%
scaled_dfprepro['cluster'] = dbscan.labels_

df_xminmax['cluster'] = dbscan.labels_

df_xminmax['cluster']=df_xminmax['cluster'].map({0:'First', 1:'Second', 2:'Third',3:'Fourth', 4:'Fifth', 5:'Sixth',6:'Seventh', 7:'Eighth', 8:'Nineth',9:'Tenth', 10:'Eleventh', -1:'Noise'})

# %%
scaled_dfprepro['cluster'] = dbscan.labels_

df_xminmax['cluster'] = dbscan.labels_

df_xminmax['cluster']=df_xminmax['cluster'].map({-1:'0',0:'1', 1:'2', 2:'3',3:'4',4:'5', 5:'6',6:'7', 7:'8', 8:'9',9:'10', 10:'11'})

df_xminmax_plot=df_xminmax.drop(["polysemy"], axis=1)

df_xminmax_plot = df_xminmax_plot.loc[df_xminmax["cluster"] != -1]


plt.figure(figsize = (8,8))


pie = df_xminmax_plot.groupby(['cluster']).size().to_frame().reset_index()
pie.rename(columns={0: 'count'}, inplace=True)

pie_labels = ['noise','first cluster', 'second cluster', 'third cluster','fourth cluster','fifth cluster','sixth cluster','seventh cluster','eighth cluster','nineth cluster','tenth cluster','eleventh cluster']
plt.pie(pie['count'], labels=pie_labels)
plt.show()

fig, axs = plt.subplots(ncols=2, nrows=6, figsize=(15,30))
for cols, x in zip(df_xminmax_plot.columns[:5], range(5)):
    for y in range(1):
        sb.boxplot(y=cols, x='cluster', data=df_xminmax_plot, order=['1', '2','3','4','5','6','7','8','9','10','11'], ax=axs[x,y], showfliers = True )

for cols, x in zip(df_xminmax_plot.columns[5:], range(5)):
    for y in range(1,2):
        sb.boxplot(y=cols, x='cluster', data=df_xminmax_plot, order=['1', '2','3','4','5','6','7','8','9','10','11'], ax=axs[x,y], showfliers = True )

# %%
print('Silhouette', silhouette_score(Y_pca_minmax, dbscan.labels_))

# %%
print('Calinski-Harabasz', metrics.calinski_harabasz_score(Y_pca_minmax, dbscan.labels_))

# %%
df_perc.isnull().sum()

df_xminmax_plot['word']=df_perc['word']
df_xminmax_plot.loc[df_xminmax['cluster']=='First']

# %%
df_xminmax_plot.loc[df_xminmax['cluster']=='Second']

# %%
df_xminmax_plot.loc[df_xminmax['cluster']=='Third']

# %%
df_xminmax_plot.loc[df_xminmax['cluster']=='Fourth']

# %%
df_xminmax_plot.loc[df_xminmax['cluster']=='Fifth']

# %% [markdown]
# ## 2.4    Analysis by hierarchical clustering

# %%
data_dist = pdist(Y_pca_minmax, metric='euclidean')
data_link = linkage(data_dist, method='complete')

res = dendrogram(data_link, truncate_mode='lastp')

# %%
hier = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='complete')
hier.fit(Y_pca_minmax)
hier.labels_
hier.n_clusters_
#hier.n_leaves_

plt.scatter( Y_pca_minmax[:,0], Y_pca_minmax[:,1],s=50, c= hier.labels_, marker='.', cmap = cm.tab20)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.show()

# %%
#Store the labels
labels = hier.labels_

#Then get the frequency count of the non-negative labels
counts = np.bincount(labels[labels>=0])

print (counts)


# %%
print('Silhouette', silhouette_score(Y_pca_minmax, hier.labels_))

# %%
print('Calinski-Harabasz', metrics.calinski_harabasz_score(Y_pca_minmax, hier.labels_))

# %%
scaled_dfprepro['cluster'] = hier.labels_

df_xminmax['cluster'] = hier.labels_

df_xminmax['cluster']=df_xminmax['cluster'].map({0:'1', 1:'2', 2:'3',3:'4', 4:'5', 5:'6'})

df_xminmax_plot=df_xminmax.drop(["polysemy"], axis=1)

plt.figure(figsize = (8,8))
pie = scaled_dfprepro.groupby(['cluster']).size().to_frame().reset_index()
pie.rename(columns={0: 'count'}, inplace=True)

pie_labels = ['first cluster', 'second cluster', 'third cluster','fourth cluster', 'fifth cluster', 'sixth cluster']
plt.pie(pie['count'], labels=pie_labels)
plt.show()
sb.set(font_scale=3.7)


fig, axs = plt.subplots(ncols=2, nrows=6, figsize=(30,60))
for cols, x in zip(df_xminmax_plot.columns[:5], range(5)):
    for y in range(1):
        g1=sb.boxplot(y=cols, x='cluster', data=df_xminmax_plot, order=['1', '2','3','4', '5','6'], ax=axs[x,y], showfliers = True )
        g1.set(xlabel=None)


for cols, x in zip(df_xminmax_plot.columns[5:], range(5)):
    for y in range(1,2):
        g2=sb.boxplot(y=cols, x='cluster', data=df_xminmax_plot, order=['1', '2','3','4', '5','6'], ax=axs[x,y], showfliers = True )
        g2.set(xlabel=None)


# %%
df_perc.isnull().sum()

df_xminmax_plot['word']=df_perc['word']


# %%
df_xminmax_plot.loc[df_xminmax['cluster']=='First']


# %%
df_xminmax_plot.loc[df_xminmax['cluster']=='Second']


# %%

df_xminmax_plot.loc[df_xminmax['cluster']=='Third']



# %%
df_xminmax_plot.loc[df_xminmax['cluster']=='Fourth']

# %%
df_xminmax_plot.loc[df_xminmax['cluster']=='Fifth']

# %%
# non serve ora 
#hier = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
#hier.fit(Y_pca_minmax)
#hier.labels_


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

#ax.scatter(Y_pca_minmax[:,0], Y_pca_minmax[:,1], Y_pca_minmax[:,2], s=5, c= hier.labels_, marker='.')


#ax.plot3D(Yemo[:,1], Yemo[:,2], Yemo[:,3], c= hier.labels_, marker='.')

#plt.scatter( Yemo[:,1], Yemo[:,2], s=50, c= hier.labels_, marker='.' )



#plt.show()

# %% [markdown]
# ## 2.5   Final Discussion

# %%


# %% [markdown]
# #3 Classification

# %% [markdown]
# ## 3.1 Classification by Decision Trees

# %%
# per il decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# visualizzarlo
from sklearn import tree
import pydotplus 
from IPython.display import Image 

# evaluazione
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score

# hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# cross-validation
from sklearn.model_selection import cross_val_score

# %%
df_class= dfprepro.copy()
df_class.head()

# %%


# %%
attributes = [col for col in df_class.columns if col != 'polysemy']
X = df_class[attributes].values
y = df_class['polysemy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=100)

len(df_class), X_train.shape[0], X_test.shape[0]

# %%
X_train.shape, X_test.shape

# %%
clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1)
clf.fit(X_train, y_train)

# %%
for col, imp in zip(attributes, clf.feature_importances_): print(col, imp)

# %%
dot_data = tree.export_graphviz (clf, out_file=None,
                                feature_names=attributes,
                                class_names=[str(v) for v in clf.classes_],
                                filled=True, rounded=True,
                                special_characters=True,
                                max_depth=2)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# %%
# apply decision tree to train set
y_pred = clf.predict(X_train)
y_pred[:5]

# %%
y_train.values[:5]

# %%
print('Accuracy', accuracy_score(y_train, y_pred))
print('F1', f1_score(y_train, y_pred, average=None))

# %%
print( classification_report(y_train, y_pred) )

# %%
# Confusion matrix for trainset
# TP, FN, FP, TN
confusion_matrix(y_train, y_pred)

# %%
# apply decision tree to test set
y_pred = clf.predict(X_test)

# %%
print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)

# %%
y_score = clf.predict_proba(X_test)


# %%
fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

# %%
plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# ## Preprocess

# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score

# %%
dfprepro.head()

# %%
df_class_ref = dfprepro.copy()
#dataframe = [df_class_ref]
#for dataset in dataframe:
#    dataset.loc[(dataset["aoa"] > 1) & (dataset["aoa"] <= 2), "aoa"] = 1
#    dataset.loc[(dataset["aoa"] > 2)& (dataset["aoa"] <= 3), "aoa"] = 2
#    dataset.loc[(dataset["aoa"] > 3)& (dataset["aoa"] <= 4), "aoa"] = 3
#    dataset.loc[(dataset["aoa"] > 4)& (dataset["aoa"] <= 5), "aoa"] = 4
#    dataset.loc[(dataset["aoa"] > 5)& (dataset["aoa"] <= 6), "aoa"] = 5
#    dataset.loc[(dataset["aoa"] > 6)&( dataset["aoa"] <= 7), "aoa"] = 6
#    dataset.loc[(dataset["aoa"] > 7), "aoa"] = 7
#df_class_ref.head()

# %%
var_to_scale=['aoa',"arousal","valence","dominance","familiarity","semsize","masculinity","perceivability"]

features = df_class_ref[var_to_scale]
scaler = MinMaxScaler().fit(features.values)
features = scaler.transform(features.values)

df_class_ref[var_to_scale] = features
df_class_ref.head()

# %% [markdown]
# ### Arousal

# %%
refvar="arousal"
taglio=0.55

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42)

clf_dt = clf_dt.fit(X_train, y_train)

# %%
plt.figure(figsize=(15,7.5))

from sklearn.tree import plot_tree
plot_tree(clf_dt,
          filled=True,
          rounded=True,
          class_names=["not aroused","aroused"],
          feature_names=X.columns)

# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score

plot_confusion_matrix(clf_dt, X_test, y_test, display_labels=["not aroused","aroused"])

# %%
y_pred = clf_dt.predict(X_train)
y_pred = clf_dt.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

y_score = clf_dt.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %%
path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts=[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
  clf_dt.fit(X_train, y_train)
  clf_dts.append(clf_dt)

# %%
train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax =plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores, marker ='o',label='train',drawstyle='steps-post')

ax.plot(ccp_alphas,test_scores, marker ='o',label='test',drawstyle='steps-post')
ax.legend()
plt.show()

# %%
clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42, ccp_alpha=0.003)

scores= cross_val_score(clf_dt,X_train,y_train, cv=10)

df=pd.DataFrame(data={'tree':range(10), 'accuracy':scores})

df.plot(x='tree', y='accuracy',marker='o',linestyle='--')


# %%
alpha_loop_values =[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
  scores= cross_val_score(clf_dt,X_train,y_train, cv=10)
  alpha_loop_values.append([ccp_alpha,np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                               columns=['alpha','mean_accuracy','std'])
  
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   marker='o',
                   linestyle='--')


# %%
alpha_results[(alpha_results['alpha']>0.0025)
&
(alpha_results['alpha']<0.005)]

# %%
ideal_ccp_alpha = 0.003242

# %%
clf_dt_pruned = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42, ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)


# %%
plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=['not aroused','aroused'])

# %%
plt.figure(figsize=(15,7.5))

from sklearn.tree import plot_tree
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["not aroused","aroused"],
          feature_names=X.columns)

# %%
y_pred = clf_dt_pruned.predict(X_train)
y_pred = clf_dt_pruned.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred,average='weighted'))
print(classification_report(y_test, y_pred))

y_score = clf_dt_pruned.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# ### Valence

# %%
refvar="valence"
taglio=0.67

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42)

clf_dt = clf_dt.fit(X_train, y_train)

# %%
path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts=[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
  clf_dt.fit(X_train, y_train)
  clf_dts.append(clf_dt)

train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax =plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores, marker ='o',label='train',drawstyle='steps-post')

ax.plot(ccp_alphas,test_scores, marker ='o',label='test',drawstyle='steps-post')
ax.legend()
plt.show()

# %%
alpha_loop_values =[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=0, ccp_alpha=ccp_alpha)
  scores= cross_val_score(clf_dt,X_train,y_train, cv=10)
  alpha_loop_values.append([ccp_alpha,np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                               columns=['alpha','mean_accuracy','std'])
  
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   marker='o',
                   linestyle='--')


# %%
alpha_results[(alpha_results['alpha']>0.0018)
&
(alpha_results['alpha']<0.0028)]

# %%
ideal_ccp_alpha = 0.002084

ideal_ccp_alpha = float(ideal_ccp_alpha)

clf_dt_pruned = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42, ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plt.figure(figsize=(15,7.5))

from sklearn.tree import plot_tree
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["not val","val"],
          feature_names=X.columns)


# %%
y_pred = clf_dt_pruned.predict(X_train)
y_pred = clf_dt_pruned.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred,average='weighted'))
print(classification_report(y_test, y_pred))

y_score = clf_dt_pruned.predict_proba(X_test)

plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=['not val','val'])


fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %%


# %% [markdown]
# ### Dominance
# 
# 

# %%
refvar="dominance"
taglio=0.57

# %%
X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42)

clf_dt = clf_dt.fit(X_train, y_train)

path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts=[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
  clf_dt.fit(X_train, y_train)
  clf_dts.append(clf_dt)

train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax =plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores, marker ='o',label='train',drawstyle='steps-post')

ax.plot(ccp_alphas,test_scores, marker ='o',label='test',drawstyle='steps-post')
ax.legend()
plt.show()

alpha_loop_values =[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
  scores= cross_val_score(clf_dt,X_train,y_train, cv=10)
  alpha_loop_values.append([ccp_alpha,np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                               columns=['alpha','mean_accuracy','std'])
  
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr='std',
                   marker='o',
                   linestyle='--')


# %%
alpha_results[(alpha_results['alpha']>0.0018)
&
(alpha_results['alpha']<0.0041)]

# %%
ideal_ccp_alpha = 0.001963

ideal_ccp_alpha = float(ideal_ccp_alpha)

clf_dt_pruned = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42, ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=['not dominant','dominant'])

plt.figure(figsize=(15,7.5))

from sklearn.tree import plot_tree
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["not dominant","dominant"],
          feature_names=X.columns)

y_pred = clf_dt_pruned.predict(X_train)
y_pred = clf_dt_pruned.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

y_score = clf_dt_pruned.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# ### Familiarity

# %%
refvar="familiarity"
taglio=0.6

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42)

clf_dt = clf_dt.fit(X_train, y_train)

path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts=[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
  clf_dt.fit(X_train, y_train)
  clf_dts.append(clf_dt)

train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax =plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores, marker ='o',label='train',drawstyle='steps-post')

ax.plot(ccp_alphas,test_scores, marker ='o',label='test',drawstyle='steps-post')
ax.legend()
plt.show()

alpha_loop_values =[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=0, ccp_alpha=ccp_alpha)
  scores= cross_val_score(clf_dt,X_train,y_train, cv=10)
  alpha_loop_values.append([ccp_alpha,np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                               columns=['alpha','mean_accuracy','std'])
  
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   marker='o',
                   linestyle='--')


# %%
alpha_results[(alpha_results['alpha']>0.0009)
&
(alpha_results['alpha']<0.0011)]

# %%
ideal_ccp_alpha = 0.000977

ideal_ccp_alpha = float(ideal_ccp_alpha)

clf_dt_pruned = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42, ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=['not dominant','dominant'])

plt.figure(figsize=(15,7.5))

from sklearn.tree import plot_tree
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["not valueable","valueable"],
          feature_names=X.columns)

y_pred = clf_dt_pruned.predict(X_train)
y_pred = clf_dt_pruned.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

y_score = clf_dt_pruned.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# ### Semsize

# %%
refvar="semsize"
taglio=0.63

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42)

clf_dt = clf_dt.fit(X_train, y_train)

path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts=[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
  clf_dt.fit(X_train, y_train)
  clf_dts.append(clf_dt)

train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax =plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores, marker ='o',label='train',drawstyle='steps-post')

ax.plot(ccp_alphas,test_scores, marker ='o',label='test',drawstyle='steps-post')
ax.legend()
plt.show()

alpha_loop_values =[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=0, ccp_alpha=ccp_alpha)
  scores= cross_val_score(clf_dt,X_train,y_train, cv=10)
  alpha_loop_values.append([ccp_alpha,np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                               columns=['alpha','mean_accuracy','std'])
  
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   marker='o',
                   linestyle='--')


# %%

alpha_results[(alpha_results['alpha']>0.0015)
&
(alpha_results['alpha']<0.002)]


# %%
ideal_ccp_alpha = 0.001601

ideal_ccp_alpha = float(ideal_ccp_alpha)

clf_dt_pruned = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42, ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=['small','big'])

plt.figure(figsize=(15,7.5))

from sklearn.tree import plot_tree
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["small","big"],
          feature_names=X.columns)

y_pred = clf_dt_pruned.predict(X_train)
y_pred = clf_dt_pruned.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

y_score = clf_dt_pruned.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# ### Masculinity

# %%
refvar="masculinity"
taglio=0.6

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42)

clf_dt = clf_dt.fit(X_train, y_train)

path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts=[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
  clf_dt.fit(X_train, y_train)
  clf_dts.append(clf_dt)

train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax =plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores, marker ='o',label='train',drawstyle='steps-post')

ax.plot(ccp_alphas,test_scores, marker ='o',label='test',drawstyle='steps-post')
ax.legend()
plt.show()

alpha_loop_values =[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=0, ccp_alpha=ccp_alpha)
  scores= cross_val_score(clf_dt,X_train,y_train, cv=10)
  alpha_loop_values.append([ccp_alpha,np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                               columns=['alpha','mean_accuracy','std'])
  
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   marker='o',
                   linestyle='--')

# %%

alpha_results[(alpha_results['alpha']>0.0015)
&
(alpha_results['alpha']<0.0025)]

# %%
ideal_ccp_alpha = 0.001985

ideal_ccp_alpha = float(ideal_ccp_alpha)

clf_dt_pruned = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42, ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=['feminine','masculine'])

plt.figure(figsize=(15,7.5))

from sklearn.tree import plot_tree
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["feminine","masculine"],
          feature_names=X.columns)

y_pred = clf_dt_pruned.predict(X_train)
y_pred = clf_dt_pruned.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

y_score = clf_dt_pruned.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# ### Polysemy

# %%
refvar="polysemy"
taglio=0.63

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42)

clf_dt = clf_dt.fit(X_train, y_train)

path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts=[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
  clf_dt.fit(X_train, y_train)
  clf_dts.append(clf_dt)

train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax =plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores, marker ='o',label='train',drawstyle='steps-post')

ax.plot(ccp_alphas,test_scores, marker ='o',label='test',drawstyle='steps-post')
ax.legend()
plt.show()

alpha_loop_values =[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=0, ccp_alpha=ccp_alpha)
  scores= cross_val_score(clf_dt,X_train,y_train, cv=10)
  alpha_loop_values.append([ccp_alpha,np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                               columns=['alpha','mean_accuracy','std'])
  
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   marker='o',
                   linestyle='--')


# %%

alpha_results[(alpha_results['alpha']>0.001)
&
(alpha_results['alpha']<0.004)]

# %%
ideal_ccp_alpha = 0.001944

ideal_ccp_alpha = float(ideal_ccp_alpha)

clf_dt_pruned = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42, ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=['not pol','pol'])

plt.figure(figsize=(15,7.5))

from sklearn.tree import plot_tree
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["not pol","pol"],
          feature_names=X.columns)

y_pred = clf_dt_pruned.predict(X_train)
y_pred = clf_dt_pruned.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred,average='weighted'))
print(classification_report(y_test, y_pred))

y_score = clf_dt_pruned.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# ### Perceivability

# %%
refvar="perceivability"
taglio=0.8

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42)

clf_dt = clf_dt.fit(X_train, y_train)

path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts=[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
  clf_dt.fit(X_train, y_train)
  clf_dts.append(clf_dt)

train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax =plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores, marker ='o',label='train',drawstyle='steps-post')

ax.plot(ccp_alphas,test_scores, marker ='o',label='test',drawstyle='steps-post')
ax.legend()
plt.show()

alpha_loop_values =[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=0, ccp_alpha=ccp_alpha)
  scores= cross_val_score(clf_dt,X_train,y_train, cv=10)
  alpha_loop_values.append([ccp_alpha,np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                               columns=['alpha','mean_accuracy','std'])
  
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   marker='o',
                   linestyle='--')


# %%
alpha_results[(alpha_results['alpha']>0.0014)
&
(alpha_results['alpha']<0.002)]


# %%
ideal_ccp_alpha = 0.001499

ideal_ccp_alpha = float(ideal_ccp_alpha)

clf_dt_pruned = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42, ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=['not peveivable','perveivable'])

plt.figure(figsize=(15,7.5))

from sklearn.tree import plot_tree
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["not perceivable","perceivable"],
          feature_names=X.columns)

y_pred = clf_dt_pruned.predict(X_train)
y_pred = clf_dt_pruned.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

y_score = clf_dt_pruned.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# ### Age of Aquisition (multi split)

# %%
refvar="aoa"

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42)

clf_dt = clf_dt.fit(X_train, y_train)

path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts=[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
  clf_dt.fit(X_train, y_train)
  clf_dts.append(clf_dt)

train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax =plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores, marker ='o',label='train',drawstyle='steps-post')

ax.plot(ccp_alphas,test_scores, marker ='o',label='test',drawstyle='steps-post')
ax.legend()
plt.show()

alpha_loop_values =[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=0, ccp_alpha=ccp_alpha)
  scores= cross_val_score(clf_dt,X_train,y_train, cv=10)
  alpha_loop_values.append([ccp_alpha,np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                               columns=['alpha','mean_accuracy','std'])
  
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   marker='o',
                   linestyle='--')


# %%
alpha_results[(alpha_results['alpha']>0.0018)
&
(alpha_results['alpha']<0.0025)]

# %%
ideal_ccp_alpha = 0.001858

ideal_ccp_alpha = float(ideal_ccp_alpha)

clf_dt_pruned = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42, ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=['0-2','2-4','4-6','6-8','8-10','10-12'])

plt.figure(figsize=(15,7.5))

from sklearn.tree import plot_tree
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=['0-2','2-4','4-6','6-8','8-10','10-12'],
          feature_names=X.columns)

y_pred = clf_dt_pruned.predict(X_train)
y_pred = clf_dt_pruned.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred,average=None))
print(classification_report(y_test, y_pred))

print(clf_dt_pruned.predict_proba(X_test))

report = classification_report(y_test, y_pred, output_dict=True)

export = pd.DataFrame(report).transpose()

print(export.to_latex())

plt.show()

# %% [markdown]
# ### Age of Aquisition (binary)

# %%
refvar="aoa"
taglio=0.6

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0




X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42)

clf_dt = clf_dt.fit(X_train, y_train)

path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts=[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
  clf_dt.fit(X_train, y_train)
  clf_dts.append(clf_dt)

train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax =plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores, marker ='o',label='train',drawstyle='steps-post')

ax.plot(ccp_alphas,test_scores, marker ='o',label='test',drawstyle='steps-post')
ax.legend()
plt.show()

alpha_loop_values =[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=0, ccp_alpha=ccp_alpha)
  scores= cross_val_score(clf_dt,X_train,y_train, cv=10)
  alpha_loop_values.append([ccp_alpha,np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                               columns=['alpha','mean_accuracy','std'])
  
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   marker='o',
                   linestyle='-')

# %%
alpha_results[(alpha_results['alpha']>0.0025)
&
(alpha_results['alpha']<0.005)]


# %%
ideal_ccp_alpha = 0.003224

ideal_ccp_alpha = float(ideal_ccp_alpha)

clf_dt_pruned = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42, ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=['younger','older'])

plt.figure(figsize=(15,7.5))

from sklearn.tree import plot_tree
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["younger","older"],
          feature_names=X.columns)

y_pred = clf_dt_pruned.predict(X_train)
y_pred = clf_dt_pruned.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

y_score = clf_dt_pruned.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# ### Web Corpus Frequency

# %%
refvar="web_corpus_freq"

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42)

clf_dt = clf_dt.fit(X_train, y_train)

path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts=[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
  clf_dt.fit(X_train, y_train)
  clf_dts.append(clf_dt)

train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax =plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores, marker ='o',label='train',drawstyle='steps-post')

ax.plot(ccp_alphas,test_scores, marker ='o',label='test',drawstyle='steps-post')
ax.legend()
plt.show()

alpha_loop_values =[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=0, ccp_alpha=ccp_alpha)
  scores= cross_val_score(clf_dt,X_train,y_train, cv=10)
  alpha_loop_values.append([ccp_alpha,np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                               columns=['alpha','mean_accuracy','std'])
  
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr='std',
                   marker='o',
                   linestyle='--')


# %%
alpha_results[(alpha_results['alpha']>0.0013)
&
(alpha_results['alpha']<0.0016)]

# %%
ideal_ccp_alpha = 0.001376

ideal_ccp_alpha = float(ideal_ccp_alpha)

clf_dt_pruned = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42, ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=['4','5','6','7','8','9'],
                      )

plt.figure(figsize=(15,7.5))

from sklearn.tree import plot_tree
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=['4','5','6','7','8','9'],
          feature_names=X.columns)

y_pred = clf_dt_pruned.predict(X_train)
y_pred = clf_dt_pruned.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred,average=None))
print(classification_report(y_test, y_pred))

print(clf_dt_pruned.predict_proba(X_test))

report = classification_report(y_test, y_pred, output_dict=True)

export = pd.DataFrame(report).transpose()

print(export.to_latex())

# %% [markdown]
# ### 3.1.1 Aoa class

# %%
df_class_aoa= dfprepro.copy()
df_class_aoa.head()

# %%
dataframe = [df_class_aoa]

# %%
for dataset in dataframe:
    dataset.loc[(dataset["aoa"] > 1) & (dataset["aoa"] <= 2), "aoa"] = 1
    dataset.loc[(dataset["aoa"] > 2)& (dataset["aoa"] <= 3), "aoa"] = 2
    dataset.loc[(dataset["aoa"] > 3)& (dataset["aoa"] <= 4), "aoa"] = 3
    dataset.loc[(dataset["aoa"] > 4)& (dataset["aoa"] <= 5), "aoa"] = 4
    dataset.loc[(dataset["aoa"] > 5)& (dataset["aoa"] <= 6), "aoa"] = 5
    dataset.loc[(dataset["aoa"] > 6)&( dataset["aoa"] <= 7), "aoa"] = 6
    dataset.loc[(dataset["aoa"] > 7), "aoa"] = 7
df_class_aoa.head()

# %%
attributes = [col for col in df_class_aoa.columns if col != 'aoa']
X = df_class_aoa[attributes].values
y = df_class_aoa['aoa']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=100)

len(df_class_aoa), X_train.shape[0], X_test.shape[0]

# %%
X=df_class_aoa.drop("aoa",axis=1).copy()
X.dtypes

# %%
y_encoded=pd.get_dummies(y,columns=["aoa"]).head()

# %%
y_encoded.dtypes

# %%
X_train.shape, X_test.shape

# %%
clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1)
clf.fit(X_train, y_train)

# %%
for col, imp in zip(attributes, clf.feature_importances_): print(col, imp)

# %%
dot_data = tree.export_graphviz (clf, out_file=None,
                                feature_names=attributes,
                                class_names=[str(v) for v in clf.classes_],
                                filled=True, rounded=True,
                                special_characters=True,
                                max_depth=2)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# %%
# apply decision tree to train set
y_pred = clf.predict(X_train)
y_pred[:5]

# %%
y_train.values[:5]

# %%
print('Accuracy', accuracy_score(y_train, y_pred))
print('F1', f1_score(y_train, y_pred, average=None))

# %%
print( classification_report(y_train, y_pred) )

# %%
# Confusion matrix for trainset
# TP, FN, FP, TN
confusion_matrix(y_train, y_pred)

# %%
# apply decision tree to test set
y_pred = clf.predict(X_test)

# %%
print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)

# %%
y_score = clf.predict_proba(X_test)

# %%
plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# ### Length

# %%
refvar="length"
taglio=0.35

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42)

clf_dt = clf_dt.fit(X_train, y_train)

path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts=[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
  clf_dt.fit(X_train, y_train)
  clf_dts.append(clf_dt)

train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax =plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores, marker ='o',label='train',drawstyle='steps-post')

ax.plot(ccp_alphas,test_scores, marker ='o',label='test',drawstyle='steps-post')
ax.legend()
plt.show()

alpha_loop_values =[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=0, ccp_alpha=ccp_alpha)
  scores= cross_val_score(clf_dt,X_train,y_train, cv=10)
  alpha_loop_values.append([ccp_alpha,np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                               columns=['alpha','mean_accuracy','std'])
  
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   marker='o',
                   linestyle='--')


# %%
alpha_results[(alpha_results['alpha']>0.001)
&
(alpha_results['alpha']<0.0025)]


# %%
ideal_ccp_alpha = 0.001538

ideal_ccp_alpha = float(ideal_ccp_alpha)

clf_dt_pruned = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42, ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=['short','long'])

plt.figure(figsize=(15,7.5))

from sklearn.tree import plot_tree
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["short","long"],
          feature_names=X.columns)

y_pred = clf_dt_pruned.predict(X_train)
y_pred = clf_dt_pruned.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

y_score = clf_dt_pruned.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# ## 3.2 Classification by KNN

# %% [markdown]
# ### Age of Aquisition

# %% [markdown]
# #### choice of k

# %%
from sklearn.neighbors import KNeighborsClassifier


refvar="aoa"
taglio=0.6

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

k = 4  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
Pred_y = neigh.predict(X_test)


error_rate = []
for i in range(1,100):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(X_train,y_train)
 pred_i = knn.predict(X_test)
 error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,100),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))

# %%

acc = []
# Will take some time
from sklearn import metrics
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))



# %%
from sklearn.neighbors import KNeighborsClassifier


refvar="aoa"
taglio=0.6

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_knn = KNeighborsClassifier(n_neighbors=24)
clf_knn.fit(X, y)

# %%
# apply KNN to train set
y_pred = clf_knn.predict(X_train)
y_pred[:5]

# %%
y_train.values[:5]

# %%
print('Accuracy', accuracy_score(y_train, y_pred))
print('F1', f1_score(y_train, y_pred, average='weighted'))
print( classification_report(y_train, y_pred) )

# %%
# Confusion matrix for trainset
# TP, FN, FP, TN
confusion_matrix(y_train, y_pred)

# %%
# apply KNN to test set
y_pred = clf_knn.predict(X_test)

# %%
print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average='weighted'))
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

plot_confusion_matrix(clf_knn,
                      X_test,
                      y_test,
                      display_labels=['younger','older'])


# %%
y_score = clf_knn.predict_proba(X_test)

# %%
fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

# %%
plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# ### Valence

# %% [markdown]
# #### choice of k

# %%
from sklearn.neighbors import KNeighborsClassifier


refvar="valence"
taglio=0.67

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

k = 4  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
Pred_y = neigh.predict(X_test)


error_rate = []
for i in range(1,100):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(X_train,y_train)
 pred_i = knn.predict(X_test)
 error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,100),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))

# %%

acc = []
# Will take some time
from sklearn import metrics
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))



# %%

refvar="valence"
taglio=0.67

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_knn = KNeighborsClassifier(n_neighbors=12)
clf_knn.fit(X, y)

y_pred = clf_knn.predict(X_train)
print('Accuracy', accuracy_score(y_train, y_pred))
print('F1', f1_score(y_train, y_pred, average='weighted'))
print( classification_report(y_train, y_pred) )

confusion_matrix(y_train, y_pred)


# %%

y_pred = clf_knn.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average='weighted'))
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

plot_confusion_matrix(clf_knn,
                      X_test,
                      y_test,
                      display_labels=['not valuable','valuable'])

y_score = clf_knn.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)


plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# ### Polysemy

# %% [markdown]
# #### choice of k

# %%
from sklearn.neighbors import KNeighborsClassifier


refvar="polysemy"
taglio=0.6

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

k = 4  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
Pred_y = neigh.predict(X_test)


error_rate = []
for i in range(1,100):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(X_train,y_train)
 pred_i = knn.predict(X_test)
 error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,100),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))

# %%

acc = []
# Will take some time
from sklearn import metrics
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.average_precision_score(y_test, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('F1 score vs. K Value')
plt.xlabel('K')
plt.ylabel('F1 Score')
print("Maximum F1:-",max(acc),"at K =",acc.index(max(acc)))



# %%

refvar="polysemy"
taglio=0.67

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_knn = KNeighborsClassifier(n_neighbors=6)
clf_knn.fit(X, y)

y_pred = clf_knn.predict(X_train)
print('Accuracy', accuracy_score(y_train, y_pred))
print('F1', f1_score(y_train, y_pred, average='weighted'))
print( classification_report(y_train, y_pred) )

confusion_matrix(y_train, y_pred)


# %%
y_pred = clf_knn.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average='weighted'))
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

plot_confusion_matrix(clf_knn,
                      X_test,
                      y_test,
                      display_labels=['not polysemic','polysemic'])

y_score = clf_knn.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)


plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# ### Age of Acquisition (multi)

# %% [markdown]
# #### choice of k

# %%
from sklearn.neighbors import KNeighborsClassifier


refvar="aoa"

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

k = 4  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
Pred_y = neigh.predict(X_test)


error_rate = []
for i in range(1,100):
 knn = KNeighborsClassifier(n_neighbors=i)
 knn.fit(X_train,y_train)
 pred_i = knn.predict(X_test)
 error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,100),error_rate,color='blue', linestyle='dashed', 
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))

# %%
clf_knn = KNeighborsClassifier(n_neighbors=37)
clf_knn.fit(X, y)

y_pred = clf_knn.predict(X_train)
print('Accuracy', accuracy_score(y_train, y_pred))
print('F1', f1_score(y_train, y_pred, average='weighted'))
print( classification_report(y_train, y_pred) )

confusion_matrix(y_train, y_pred)


# %%
y_pred = clf_knn.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average='weighted'))
print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

plot_confusion_matrix(clf_knn,
                      X_test,
                      y_test,
                      display_labels=['0-2','2-4','4-6','6-8','8-10'])

y_score = clf_knn.predict_proba(X_test)

report = classification_report(y_test, y_pred, output_dict=True)

export = pd.DataFrame(report).transpose()

print(export.to_latex())

# %% [markdown]
# ## Random Forest

# %%
from sklearn.ensemble import RandomForestClassifier

# %% [markdown]
# ### Valence

# %%
refvar='valence'
taglio=0.67

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Instantiate model with 10 decision trees
model = RandomForestClassifier(n_estimators = 380, random_state = 42)
# Train the model on training data
ra=model.fit(X_train, y_train)


#TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE

prediction_test = model.predict(X_test)
#print(y_test, prediction_test)

#Print the prediction accuracy
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))
#Test accuracy for various test sizes and see how it gets better with more training data

#One amazing feature of Random forest is that it provides us info on feature importances
# Get numerical feature importances
#importances = list(model.feature_importances_)

#Let us print them into a nice format.

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)

y_pred = model.predict(X_train)
y_pred = model.predict(X_test)

plot_confusion_matrix(ra,
                      X_test,
                      y_test,
                      display_labels=['not val','val'],
                      )

y_score = model.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred,average='weighted'))
print(classification_report(y_test, y_pred))


# %% [markdown]
# ### Polysemy

# %%
refvar='polysemy'
taglio=0.67

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

# Instantiate model with 10 decision trees
model = RandomForestClassifier(n_estimators = 385, random_state = 42)
# Train the model on training data
ra=model.fit(X_train, y_train)


#TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE

prediction_test = model.predict(X_test)
#print(y_test, prediction_test)

#Print the prediction accuracy
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))
#Test accuracy for various test sizes and see how it gets better with more training data

#One amazing feature of Random forest is that it provides us info on feature importances
# Get numerical feature importances
#importances = list(model.feature_importances_)

#Let us print them into a nice format.

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)


y_pred = model.predict(X_train)
y_pred = model.predict(X_test)

plot_confusion_matrix(ra,
                      X_test,
                      y_test,
                      display_labels=['not pol','pol']
                      )

y_score = model.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred,average='weighted'))
print(classification_report(y_test, y_pred))

# %% [markdown]
# ### Age of Aquisition

# %%
refvar='aoa'
taglio=0.6

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Instantiate model with 10 decision trees
model = RandomForestClassifier(n_estimators = 380, random_state = 42)
# Train the model on training data
ra=model.fit(X_train, y_train)


#TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE

prediction_test = model.predict(X_test)
#print(y_test, prediction_test)

#Print the prediction accuracy
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))
#Test accuracy for various test sizes and see how it gets better with more training data

#One amazing feature of Random forest is that it provides us info on feature importances
# Get numerical feature importances
#importances = list(model.feature_importances_)

#Let us print them into a nice format.

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)


y_pred = model.predict(X_train)
y_pred = model.predict(X_test)

plot_confusion_matrix(ra,
                      X_test,
                      y_test,
                      display_labels=['younger','older']
                      )

y_score = model.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred,average='weighted'))
print(classification_report(y_test, y_pred))

# %% [markdown]
# ### Out of bag error

# %%

import matplotlib.pyplot as plt

from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

refvar='valence'
taglio=0.67

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0


# NOTE: Setting the `warm_start` construction parameter to `True` disables
# support for parallelized ensembles but is necessary for tracking the OOB
# error trajectory during training.
ensemble_clfs = [
    (
        "RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(
            warm_start=True,
            oob_score=True,
            max_features="sqrt",
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(
            warm_start=True,
            max_features="log2",
            oob_score=True,
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "RandomForestClassifier, max_features=None",
        RandomForestClassifier(
            warm_start=True,
            max_features=None,
            oob_score=True,
            random_state=RANDOM_STATE,
        ),
    ),
]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 100
max_estimators = 1000

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1, 5):
        clf.set_params(n_estimators=i)
        clf.fit(X, y)

        # Record the OOB error for each `n_estimators=i` setting.
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("OOB error rate")
plt.legend(loc="upper right")
plt.show()

# %% [markdown]
# ## Entropy (Decision Tree)

# %% [markdown]
# ### Age of Acquisition

# %%
refvar="aoa"
taglio=0.6

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0




X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42)

clf_dt = clf_dt.fit(X_train, y_train)

path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts=[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(criterion='entropy',random_state=0, ccp_alpha=ccp_alpha)
  clf_dt.fit(X_train, y_train)
  clf_dts.append(clf_dt)

train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax =plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores, marker ='o',label='train',drawstyle='steps-post')

ax.plot(ccp_alphas,test_scores, marker ='o',label='test',drawstyle='steps-post')
ax.legend()
plt.show()

alpha_loop_values =[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=0, ccp_alpha=ccp_alpha)
  scores= cross_val_score(clf_dt,X_train,y_train, cv=10)
  alpha_loop_values.append([ccp_alpha,np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                               columns=['alpha','mean_accuracy','std'])
  
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr='std',
                   marker='o',
                   linestyle='-')

# %%
alpha_results[(alpha_results['alpha']>0.002)
&
(alpha_results['alpha']<0.004)]


# %%
ideal_ccp_alpha = 0.002246

ideal_ccp_alpha = float(ideal_ccp_alpha)

clf_dt_pruned = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42, ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=['young','old'])

plt.figure(figsize=(15,7.5))

from sklearn.tree import plot_tree
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["young","old"],
          feature_names=X.columns)

y_pred = clf_dt_pruned.predict(X_train)
y_pred = clf_dt_pruned.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

y_score = clf_dt_pruned.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# ### Polysemy

# %%
refvar="polysemy"
taglio=0.6

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0




X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42)

clf_dt = clf_dt.fit(X_train, y_train)

path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts=[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(criterion='entropy',random_state=0, ccp_alpha=ccp_alpha)
  clf_dt.fit(X_train, y_train)
  clf_dts.append(clf_dt)

train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax =plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores, marker ='o',label='train',drawstyle='steps-post')

ax.plot(ccp_alphas,test_scores, marker ='o',label='test',drawstyle='steps-post')
ax.legend()
plt.show()

alpha_loop_values =[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=0, ccp_alpha=ccp_alpha)
  scores= cross_val_score(clf_dt,X_train,y_train, cv=10)
  alpha_loop_values.append([ccp_alpha,np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                               columns=['alpha','mean_accuracy','std'])
  
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr='std',
                   marker='o',
                   linestyle='-')

# %%
alpha_results[(alpha_results['alpha']>0.002)
]


# %%
ideal_ccp_alpha = 0.003002

ideal_ccp_alpha = float(ideal_ccp_alpha)

clf_dt_pruned = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42, ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=['not polysemic','polysemic'])

plt.figure(figsize=(15,7.5))

from sklearn.tree import plot_tree
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["not polysemic","polysemic"],
          feature_names=X.columns)

y_pred = clf_dt_pruned.predict(X_train)
y_pred = clf_dt_pruned.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

y_score = clf_dt_pruned.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# ### Valence

# %%
refvar="valence"
taglio=0.67

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0




X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_dt = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42)

clf_dt = clf_dt.fit(X_train, y_train)

path = clf_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts=[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
  clf_dt.fit(X_train, y_train)
  clf_dts.append(clf_dt)

train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax =plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores, marker ='o',label='train',drawstyle='steps-post')

ax.plot(ccp_alphas,test_scores, marker ='o',label='test',drawstyle='steps-post')
ax.legend()
plt.show()

alpha_loop_values =[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=0, ccp_alpha=ccp_alpha)
  scores= cross_val_score(clf_dt,X_train,y_train, cv=10)
  alpha_loop_values.append([ccp_alpha,np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                               columns=['alpha','mean_accuracy','std'])
  
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr='std',
                   marker='o',
                   linestyle='-')

# %%
alpha_results[(alpha_results['alpha']>0.0025)
&
(alpha_results['alpha']<0.0035)]


# %%
ideal_ccp_alpha = 0.003041

ideal_ccp_alpha = float(ideal_ccp_alpha)

clf_dt_pruned = DecisionTreeClassifier(criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42, ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=['not valuable','valuable'])

plt.figure(figsize=(15,7.5))

from sklearn.tree import plot_tree
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["not valuable","valuable"],
          feature_names=X.columns)

y_pred = clf_dt_pruned.predict(X_train)
y_pred = clf_dt_pruned.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

y_score = clf_dt_pruned.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# ## Citraro

# %%
###### Lecture Pipeline ######
# you can/have to modify the order for your project,
# e.g. hyperparameter tuning before evaluating a decision tree

# Hold-out
from sklearn.model_selection import train_test_split

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
# Optional: KNN
from sklearn.neighbors import KNeighborsClassifier

##### Decision Tree Visualization ###
from sklearn import tree # export_graphviz() method
import pydotplus # viz stuff
from IPython.display import Image # viz stuff
####################################

######## Evaluation #######
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
# Optional: baseline comparison
from sklearn.dummy import DummyClassifier
###########################

# hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# cross-validation
from sklearn.model_selection import cross_val_score

# %%
refvar="valence"
taglio=0.67

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0




X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)



# %%
clf = DecisionTreeClassifier(criterion='gini',
                             max_depth=None ,
                             min_samples_split=10,
                             min_samples_leaf=5,
                            )

clf.fit(X_train, y_train)

# %%
imp_dict = {attr: clf.feature_importances_[i] for i, attr in enumerate(attributes)}
imp_dict

{k: v for k, v in sorted(imp_dict.items(), key=lambda item: item[1], reverse=True)}

# %%
dot_data = tree.export_graphviz (clf, out_file=None,
                                feature_names=attributes,
                                class_names=[str(v) for v in clf.classes_],
                                filled=True, rounded=True,
                                special_characters=True,
                                max_depth=2)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

# %%
y_pred = clf.predict(X_train)

# %%
print('Accuracy', accuracy_score(y_train, y_pred))
print('F1', f1_score(y_train, y_pred, average=None))

# %%
print( classification_report(y_train, y_pred) )

# %%
# NOT SUGGESTED FOR THE PROJECT

cf = confusion_matrix(y_train, y_pred)
cf

# %%
y_pred = clf.predict( X_test )

# %%
y_score = clf.predict_proba(X_test)
y_score[:6]

# %%
fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)
plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %%
dummy_clf = DummyClassifier(strategy='constant', constant=1)
dummy_clf.fit(X_test, y_test)

y_pred = dummy_clf.predict(X_test)

# %%
param_list = {'max_depth': [None] + [2, 3, 4],
             'min_samples_split': [2, 5, 10, 20],
             'min_samples_leaf': [1, 5, 10, 20]
             }

grid_search = GridSearchCV(clf, param_grid=param_list, scoring='f1')
grid_search.fit(X, y)

res = grid_search.cv_results_

# %%
grid_search.best_estimator_

# %%
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# %%
report(res, n_top=3)

# %%

clf_dt_pruned = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=5, min_samples_leaf=20)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      y_test,
                      display_labels=['not valuable','valuable'])

plt.figure(figsize=(15,7.5))

from sklearn.tree import plot_tree
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["not valuable","valuable"],
          feature_names=X.columns)

y_pred = clf_dt_pruned.predict(X_train)
y_pred = clf_dt_pruned.predict(X_test)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))

y_score = clf_dt_pruned.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# # 4 Patter mining

# %% [markdown]
# ## 4.1 Preprocessing

# %%
dfprepro.head()

# %%
dfprepro.describe()

# %%
df_pm= dfprepro.copy()

# %%
var_to_scale=['length','aoa',"arousal","valence","dominance","familiarity","semsize","masculinity","perceivability"]

features = df_pm[var_to_scale]
scaler = MinMaxScaler().fit(features.values)
features = scaler.transform(features.values)


df_pm[var_to_scale] = 4*features-0.5
df_pm.head()

# %%
df_pm=df_pm.round()

# %%
df_pm.head()

# %%
df_pm.isnull().sum()

# %%
df_pm_copy=df_pm.copy()

# %%
df_pm.loc[(df_pm['web_corpus_freq'].isnull() == True), 'web_corpus_freq'] = df_pm['web_corpus_freq'].mean()


# %%
df_pm['length'] = df_pm['length'].astype(str) + '_Lenght'
df_pm['arousal'] = df_pm['arousal'].astype(str) + '_Arousal'
df_pm['valence'] = df_pm['valence'].astype(str) + '_Valence'
df_pm['dominance'] = df_pm['dominance'].astype(str) + '_Dominance'
df_pm['familiarity'] = df_pm['familiarity'].astype(str) + '_Familiarity'
df_pm['aoa'] = df_pm['aoa'].astype(str) + '_Age_of_Aquisition'
df_pm['semsize'] = df_pm['semsize'].astype(str) + '_SemSize'
df_pm['masculinity'] = df_pm['masculinity'].astype(str) + '_Masculinity'
df_pm['web_corpus_freq'] = df_pm['web_corpus_freq'].astype(str) + '_Web_Corpus_Freq'
df_pm['perceivability'] = df_pm['perceivability'].astype(str) + '_Perceivability'


df_pm.head()

# %%
polysemy_dict = {0: 'Not Polysemy', 1: 'Polysemy'}
df_pm['polysemy'] = df_pm['polysemy'].map(polysemy_dict)

df_pm.head()

# %% [markdown]
# ## 4.2 Patter mining

# %% [markdown]
# ### Frequent itemsets

# %%
X = df_pm.values.tolist()

# %%
X[0]

# %%
help(apriori)

# %%
itemsets = apriori(X, target='s', supp=2, zmin=2, report='S')

# %%
itemsets[:5]

# %%
# frequent itemset
itemsets[0][0]

# %%
# support
itemsets[0][1]

# %%
itemsets = apriori(X, target='m', supp=2, zmin=2, report='S')

# %%
itemsets[:5]

# %%


# %%
itemsets[0][0]

# %%
df_pm.describe()

# %%
len_max_it = []
len_cl_it = []
len_all_it = []
for i in range(1, 8+1):
    max_itemsets = apriori(X, target='m', supp=i, zmin=3)
    cl_itemsets = apriori(X, target='c', supp=i, zmin=3)
    all_itemsets = apriori(X, target='s', supp=i, zmin=3)
    len_max_it.append( len(max_itemsets)  )
    len_cl_it.append( len(cl_itemsets) )
    len_all_it.append( len(all_itemsets) )
    
plt.plot(len_max_it, label='maximal')
plt.plot(len_all_it, label='all')
plt.plot(len_cl_it, label='closed')
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('%support', fontsize=15)

plt.show()

# %%
len_max_it = []
len_cl_it = []
len_all_it = []
for i in range(1, 6+1):
    max_itemsets = apriori(X, target='m', supp=2, zmin=i)
    cl_itemsets = apriori(X, target='c', supp=2, zmin=i)
    all_itemsets = apriori(X, target='s', supp=2, zmin=i)
    len_max_it.append( len(max_itemsets)  )
    len_cl_it.append( len(cl_itemsets) )
    len_all_it.append( len(all_itemsets) )
    
plt.plot(len_max_it, label='maximal')
plt.plot(len_cl_it, label='closed')
plt.plot(len_all_it, label='all')
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('#zmin', fontsize=15)

plt.show()

# %%
df_no_pol=df_pm.drop('polysemy',axis=1)

# %%
df_pm.head()

# %%
X_no_pol = df_no_pol.values.tolist()

# %%
len_max_it = []
len_cl_it = []
len_all_it = []
for i in range(1, 5+1):
    max_itemsets = apriori(X, target='m', supp=i, zmin=3)
    cl_itemsets = apriori(X, target='c', supp=i, zmin=3)
    all_itemsets = apriori(X, target='s', supp=i, zmin=3)
    len_max_it.append( len(max_itemsets)  )
    len_cl_it.append( len(cl_itemsets) )
    len_all_it.append( len(all_itemsets) )
    
plt.plot(len_all_it, label='all')
plt.plot(len_max_it, label='maximal')
plt.plot(len_cl_it, label='closed')
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('%support', fontsize=15)

plt.show()

# %%
len_max_it = []
len_cl_it = []
len_all_it = []
for i in range(1, 6+1):
    max_itemsets = apriori(X, target='m', supp=2, zmin=i)
    cl_itemsets = apriori(X, target='c', supp=2, zmin=i)
    all_itemsets = apriori(X, target='s', supp=2, zmin=i)
    len_max_it.append( len(max_itemsets)  )
    len_cl_it.append( len(cl_itemsets) )
    len_all_it.append( len(all_itemsets) )

plt.plot(len_all_it, label='all')
plt.plot(len_max_it, label='maximal')
plt.plot(len_cl_it, label='closed')
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('%support', fontsize=15)

plt.show()

# %%
filter_0 = []
filter_1 = []


for i in range(2, 8+1):
    max_itemsets = apriori(X, target='a', supp=i, zmin=3)
    
    len_filter_0=0

    len_filter_0+=len([item for item in max_itemsets if '0.0_Age_of_Aquisition' in item[0]])
    len_filter_0+=len([item for item in max_itemsets if '1.0_Age_of_Aquisition' in item[0]])
    len_filter_0+=len([item for item in max_itemsets if '2.0_Age_of_Aquisition' in item[0]])
    len_filter_0+=len([item for item in max_itemsets if '3.0_Age_of_Aquisition' in item[0]])
    len_filter_0+=len([item for item in max_itemsets if '4.0_Age_of_Aquisition' in item[0]])
    len_filter_0+=len([item for item in max_itemsets if '5.0_Age_of_Aquisition' in item[0]])

    filter_0.append( len_filter_0 )

    len_filter_1=0
    len_filter_1+=len([item for item in max_itemsets if '6.0_Age_of_Aquisition' in item[0]])
    len_filter_1+=len([item for item in max_itemsets if '7.0_Age_of_Aquisition' in item[0]])
    len_filter_1+=len([item for item in max_itemsets if '8.0_Age_of_Aquisition' in item[0]])
    len_filter_1+=len([item for item in max_itemsets if '9.0_Age_of_Aquisition' in item[0]])
    len_filter_1+=len([item for item in max_itemsets if '10.0_Age_of_Aquisition' in item[0]])

    filter_1.append( len_filter_1 )

    
plt.plot(filter_0, label='younger')
plt.plot(filter_1, label='older')



plt.legend(fontsize=15)
plt.xticks(fontsize=20)
plt.xlabel('%support', fontsize=15)

plt.show()

# %%
filter_0 = []
filter_1 = []


for i in range(2, 8+1):
    max_itemsets = apriori(X, target='a', supp=i, zmin=3)
    
    len_filter_0=0

    len_filter_0+=len([item for item in max_itemsets if '0.0_Valence' in item[0]])
    len_filter_0+=len([item for item in max_itemsets if '1.0_Valence' in item[0]])
    len_filter_0+=len([item for item in max_itemsets if '2.0_Valence' in item[0]])
    len_filter_0+=len([item for item in max_itemsets if '3.0_Valence' in item[0]])
    len_filter_0+=len([item for item in max_itemsets if '4.0_Valence' in item[0]])
    len_filter_0+=len([item for item in max_itemsets if '5.0_Valence' in item[0]])

    filter_0.append( len_filter_0 )

    len_filter_1=0
    len_filter_1+=len([item for item in max_itemsets if '6.0_Valence' in item[0]])
    len_filter_1+=len([item for item in max_itemsets if '7.0_Valence' in item[0]])
    len_filter_1+=len([item for item in max_itemsets if '8.0_Valence' in item[0]])
    len_filter_1+=len([item for item in max_itemsets if '9.0_Valence' in item[0]])
    len_filter_1+=len([item for item in max_itemsets if '10.0_Valence' in item[0]])

    filter_1.append( len_filter_1 )

    
plt.plot(filter_0, label='no val')
plt.plot(filter_1, label='val')



plt.legend(fontsize=15)
plt.xticks(fontsize=20)
plt.xlabel('%support', fontsize=15)

plt.show()

# %% [markdown]
# ### Association Rules

# %%
rules = apriori(X_no_pol, target='r', supp=7, zmin=1, conf=75, report='aScl')
len(rules)

# %%
rules_supp = []
rules_zmin = []
rules_conf = []

supp_range=[6,8,10,12,14]

zmin_range=[1,2,3,4,5]

conf_range=[50,60,70,80,90]

for i,j,k in zip(supp_range,zmin_range,conf_range):
    suppp = apriori(X, target='r', supp=i,conf=60)
    zminn = apriori(X, target='r', zmin=j,conf=60,supp=8)
    confi = apriori(X, target='r', conf=k,supp=8)
    rules_supp.append( len(suppp)  )
    rules_zmin.append( len(zminn) )
    rules_conf.append( len(confi) )
    
plt.plot(rules_supp, label='supp')
plt.plot(rules_zmin, label='zmin')
plt.plot(rules_conf, label='conf')
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('iteration', fontsize=15)

plt.show()

# %%
count=0
for i in range(len(rules)):
  if rules[i][5]>1.6:
    count+=1
    print(rules[i][1], rules[i][0], rules[i][5])
    print("\n")
print(count)

# %%
dom1=[r for r in rules if r[0] == '1.0_Dominance']

# %%
val2=[r for r in rules if r[0] == '2.0_Valence']

# %%
print(dom1)

# %%
for el in range(len(dom1)):
  print( 'to_predict:', [r for r in rules if r[0] == '1.0_Dominance'][el][0])
  print( 'how?', [r for r in rules if r[0] == '1.0_Dominance'][el][1])

# %%
for el in range(len(val2)):
  print( 'to_predict:', [r for r in rules if r[0] == '2.0_Valence'][el][0])
  print( 'how?', [r for r in rules if r[0] == '2.0_Valence'][el][1])

# %%
# <--
rules[0][0], rules[0][1]

# %%
# asbolute support
rules[0][2]

# %%
# support as percentage
rules[0][3]

# %%
# confidence
rules[0][4]

# %%
# lift
rules[0][5]

# %%



