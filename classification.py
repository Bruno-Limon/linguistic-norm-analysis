# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec

#scaling, normalization
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn import metrics

from google.colab import files


# %%
#caricamento del dataset
df = pd.read_csv('words_glasgow.csv')
#faccio una copia del dataset in caso di manipolazione dati
dfcopy= df.copy()

# %%
df2 = df.copy()
df2["perceivability"] = df2[["imageability", "concreteness"]].mean(axis=1)
df_perc=df2.drop(["concreteness","imageability"], axis=1)
dfprepro= df_perc.copy()
dfprepro=dfprepro.rename(columns={"gender": "masculinity"})
dfprepro.loc[(dfprepro['web_corpus_freq'].isnull() == True), 'web_corpus_freq'] = dfprepro['web_corpus_freq'].mean()
dfprepro["web_corpus_log"] = pd.qcut(dfprepro["web_corpus_freq"], 10) #taglio la variabile web_corpus_freq in 10 gruppi
dataframe = [dfprepro]
for dataset in dataframe:
    dataset.loc[(dataset["web_corpus_freq"] > 10000) & (dataset["web_corpus_freq"] <= 100000), "web_corpus_freq"] = 4
    dataset.loc[(dataset["web_corpus_freq"] > 100000) & (dataset["web_corpus_freq"] <= 1000000), "web_corpus_freq"] = 5
    dataset.loc[(dataset["web_corpus_freq"] > 1000000) & (dataset["web_corpus_freq"] <= 10000000), "web_corpus_freq"] = 6
    dataset.loc[(dataset["web_corpus_freq"] > 10000000) & (dataset["web_corpus_freq"] <= 100000000), "web_corpus_freq"] = 7
    dataset.loc[(dataset["web_corpus_freq"] > 100000000) & (dataset["web_corpus_freq"] <= 1000000000), "web_corpus_freq"] = 8
    dataset.loc[dataset["web_corpus_freq"] > 1000000000, "web_corpus_freq"] = 9
dfprepro = dfprepro.drop(["web_corpus_log","word"], axis=1)


# %% [markdown]
# # Preprocess for classification

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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score

# %%


# %%
df_class_ref = dfprepro.copy()
var_to_scale=['aoa',"arousal","valence","dominance","familiarity","semsize","masculinity","perceivability"]

features = df_class_ref[var_to_scale]
scaler = MinMaxScaler().fit(features.values)
features = scaler.transform(features.values)

df_class_ref[var_to_scale] = features


# %%
df_class_ref

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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

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
alpha_results[(alpha_results['alpha']>0.001)
&
(alpha_results['alpha']<0.005)]

# %%
indexmax = alpha_results[['mean_accuracy']].idxmax()

maxalpha=alpha_results.loc[indexmax,'alpha']

ideal_ccp_alpha = float(maxalpha)

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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %%


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
indexmax = alpha_results[['mean_accuracy']].idxmax()

maxalpha=alpha_results.loc[indexmax,'alpha']

ideal_ccp_alpha = float(maxalpha)

# %%
clf_dt_pruned = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,random_state=42, ccp_alpha=ideal_ccp_alpha)

clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

plt.figure(figsize=(15,7.5))
clf_dt_pruned.classes_

from sklearn.tree import plot_tree
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=[str(v) for v in clf_dt_pruned.classes_],
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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

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

alpha_loop_values =[]

for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
  scores= cross_val_score(clf_dt,X_train,y_train, cv=10)
  alpha_loop_values.append([ccp_alpha,np.mean(scores), np.std(scores)])

alpha_results = pd.DataFrame(alpha_loop_values,
                               columns=['alpha','mean_accuracy','std'])
  
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   marker='o',
                   linestyle='--')


# %%
indexmax = alpha_results[['mean_accuracy']].idxmax()

maxalpha=alpha_results.loc[indexmax,'alpha']

ideal_ccp_alpha = float(maxalpha)

# %%
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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %%


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
indexmax = alpha_results[['mean_accuracy']].idxmax()

maxalpha=alpha_results.loc[indexmax,'alpha']

ideal_ccp_alpha = float(maxalpha)

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
          class_names=["not familiar","familiar"],
          feature_names=X.columns,
          max_depth=3,
          fontsize=7)

y_pred = clf_dt_pruned.predict(X_train)
y_pred = clf_dt_pruned.predict(X_test)
plt.savefig('plot_of_tree.pdf')

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

y_score = clf_dt_pruned.predict_proba(X_test)

fpr, tpr, th = roc_curve(y_test, y_score[:,1])

roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8,5))

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)
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
indexmax = alpha_results[['mean_accuracy']].idxmax()

maxalpha=alpha_results.loc[indexmax,'alpha']

ideal_ccp_alpha = float(maxalpha)

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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

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
indexmax = alpha_results[['mean_accuracy']].idxmax()

maxalpha=alpha_results.loc[indexmax,'alpha']

ideal_ccp_alpha = float(maxalpha)

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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

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
indexmax = alpha_results[['mean_accuracy']].idxmax()

maxalpha=alpha_results.loc[indexmax,'alpha']

ideal_ccp_alpha = float(maxalpha)

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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

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
indexmax = alpha_results[['mean_accuracy']].idxmax()

maxalpha=alpha_results.loc[indexmax,'alpha']

ideal_ccp_alpha = float(maxalpha)

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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
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
indexmax = alpha_results[['mean_accuracy']].idxmax()

maxalpha=alpha_results.loc[indexmax,'alpha']

ideal_ccp_alpha = float(maxalpha)

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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

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
indexmax = alpha_results[['mean_accuracy']].idxmax()

maxalpha=alpha_results.loc[indexmax,'alpha']

ideal_ccp_alpha = float(maxalpha)

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

refvar="aoa"
taglio=0.6

X=df_class_ref.drop(refvar,axis=1).copy()

y=df_class_ref[refvar].copy()

y_up_index = y >= taglio

y[y_up_index]=1

y_zero_index = y < taglio

y[y_zero_index]=0

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf_knn = KNeighborsClassifier(n_neighbors=error_rate.index(min(error_rate)))
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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

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

clf_knn = KNeighborsClassifier(n_neighbors=error_rate.index(min(error_rate)))
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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

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

clf_knn = KNeighborsClassifier(n_neighbors=acc.index(max(acc)))
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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

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
indexmax = alpha_results[['mean_accuracy']].idxmax()

maxalpha=alpha_results.loc[indexmax,'alpha']

ideal_ccp_alpha = float(maxalpha)
print(ideal_ccp_alpha)


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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

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
indexmax = alpha_results[['mean_accuracy']].idxmax()

maxalpha=alpha_results.loc[indexmax,'alpha']

ideal_ccp_alpha = float(maxalpha)
print(ideal_ccp_alpha)

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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

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
indexmax = alpha_results[['mean_accuracy']].idxmax()

maxalpha=alpha_results.loc[indexmax,'alpha']

ideal_ccp_alpha = float(maxalpha)

print(ideal_ccp_alpha)

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

plt.plot(fpr, tpr, label='$AUC$ = %.3f' % (roc_auc))
plt.legend(loc="lower right", fontsize=14, frameon=False)

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)

plt.tick_params(axis='both', which='major', labelsize=22)
plt.show()

# %% [markdown]
# # Multisplit target

# %%
df_class_ref = dfprepro.copy()
dataframe = [df_class_ref]
for dataset in dataframe:
    dataset.loc[(dataset["aoa"] > 1) & (dataset["aoa"] <= 2), "aoa"] = 1
    dataset.loc[(dataset["aoa"] > 2)& (dataset["aoa"] <= 3), "aoa"] = 2
    dataset.loc[(dataset["aoa"] > 3)& (dataset["aoa"] <= 4), "aoa"] = 3
    dataset.loc[(dataset["aoa"] > 4)& (dataset["aoa"] <= 5), "aoa"] = 4
    dataset.loc[(dataset["aoa"] > 5)& (dataset["aoa"] <= 6), "aoa"] = 5
    dataset.loc[(dataset["aoa"] > 6)&( dataset["aoa"] <= 7), "aoa"] = 6
    dataset.loc[(dataset["aoa"] > 7), "aoa"] = 7
df_class_ref.head()

# %%
var_to_scale=["arousal","valence","dominance","familiarity","semsize","masculinity","perceivability"]

features = df_class_ref[var_to_scale]
scaler = MinMaxScaler().fit(features.values)
features = scaler.transform(features.values)

df_class_ref[var_to_scale] = features
df_class_ref.head()

# %%


# %% [markdown]
# ### Age of Aquisition (DT)

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
alpha_results[(alpha_results['alpha']>0.0015)
&
(alpha_results['alpha']<0.0022)]

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
# ### Age of Acquisition (KNN)

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
clf_knn = KNeighborsClassifier(n_neighbors=14)
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


