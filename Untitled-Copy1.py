
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load the data
data=pd.read_csv('creditcard.csv')
print(data.shape)


# In[2]:


data=data.sample(frac=0.1,random_state=1)
print(data.shape)


# In[3]:


columns=data.columns.tolist()
columns=[c for c in columns if c not in 'Class']


# In[4]:


print(columns.shape)


# In[6]:


columns=data.columns.tolist()
columns=[c for c in columns if c not in ["Class"]]
target="Class"
X=data[columns]
Y=data[target]
print(X.shape)
print(Y.shape)
#print(columns)


# In[7]:


print(columns)


# In[8]:


print(data.columns)


# In[15]:


Fraud=data[data["Class"]==1]
Valid=data[data["Class"]==0]
outliner_fraction=len(Fraud)/float(len(Valid))
print('fraud: {}'.format(len(Fraud)))
print('valid: {}'.format(len(Valid)))
print(outliner_fraction)


# In[16]:


from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest


# In[17]:


state=1
classifiers={
    "IsolationForest":IsolationForest(max_samples=len(X),contamination=outliner_fraction,random_state=state)
}
n_outliners=len(Fraud)
for i,(clf_name,clf) in enumerate(classifiers.items()):
    if clf_name=="IsolationForest":
        clf.fit(X)
        scores_pred=clf.decision_function(X)
        y_pred=clf.predict(X)
        
        y_pred[y_pred==1]=0
        y_pred[y_pred==-1]=1
        n_errors=y_pred[y_pred!=Y].sum()
        print('{}: {}'.format(clf_name,n_errors))
        print(accuracy_score(Y,y_pred))
        print(classification_report(Y,y_pred))

