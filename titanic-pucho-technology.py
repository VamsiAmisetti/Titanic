#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.style as style
style.available

import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV


# In[3]:


df = pd.read_csv('train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('test.csv' , header = 0, dtype={'Age': np.float64})


# In[4]:


df.head()


# In[5]:


test.head()


# In[6]:


df.groupby(by=['Pclass'])['Survived'].agg(['mean','count'])


# In[7]:


sex_survived= df.groupby(by=['Sex','Survived'])['Survived'].agg(['count']).reset_index()
sex_survived


# In[8]:


plt.figure(figsize=(10, 5))
style.use('seaborn-notebook')
sns.barplot(data=sex_survived, x='Sex',y='count', hue='Survived');


# In[9]:


df_all = [df,test]


# In[10]:


for data in df_all:
    print("\n -------- {data.index } ------- \n")
    print(data.isnull().sum())


# In[11]:


for data in df_all:
    data['isAlone']=1

    data['Family_No'] = data['Parch'] + data['SibSp'] + 1
        
    data['isAlone'].loc[data['Family_No']>1]=0
    
    data['Age'].fillna(data['Age'].mean(), inplace=True)

    data['Embarked'].fillna(data['Embarked'].mode().iloc[0], inplace=True)
    
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)
    


# In[12]:


test.isAlone.value_counts()


# In[13]:


for data in df_all:
    data.drop(columns=['PassengerId','Name','Cabin','Ticket','SibSp','Parch'],inplace=True,axis=1)


# In[14]:


for data in df_all:
    print("\n -------- {data.index } ------- \n")
    print(data.isnull().sum())


# In[15]:


test = pd.get_dummies(test,columns=['Sex','Embarked'])
df = pd.get_dummies(df,columns=['Sex','Embarked'])


# In[16]:


df.head()


# In[17]:


y=df['Survived']
X=df.drop(columns=['Survived'],axis=1)


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


# In[19]:


DT= DecisionTreeClassifier()
DT.fit(X_train, y_train)
DT.score(X_test,y_test)


# In[20]:


parameters1 = [{'max_depth':np.linspace(1, 15, 15),'min_samples_split': np.linspace(0.1, 1.0, 5, endpoint=True)}]


# In[21]:


from sklearn.model_selection import GridSearchCV

Grid1 = GridSearchCV(DT, parameters1, cv=4,return_train_score=True)
Grid1.fit(X_train,y_train)


# In[22]:


scores = Grid1.cv_results_


# In[23]:


for param, mean_train in zip(scores['params'],scores['mean_train_score']):
    print("{param} accuracy on training data is {mean_train}")


# In[24]:


Grid1.best_estimator_


# In[25]:


max(scores['mean_train_score'])


# In[26]:


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# In[27]:


RF = RandomForestClassifier()
XGB = XGBClassifier()


# In[28]:


parameters2 = [{'max_depth':np.linspace(1, 15, 15),'min_samples_split': np.linspace(0.1, 1.0, 5, endpoint=True)}]

parameters3 =[{"learning_rate": [0.05, 0.10, 0.15, 0.20] ,"max_depth": [ 3, 4, 5, 6, 8, 10], "min_child_weight": [3,5,7],"gamma": [ 0.0, 0.1, 0.2 ,0.3],"colsample_bytree" : [ 0.4, 0.5]}]


# In[29]:


Grid1 = GridSearchCV(XGB, parameters3, cv=3,return_train_score=True)

Grid1.fit(X_train,y_train)


# In[30]:


scores = Grid1.cv_results_


# In[31]:


Grid1.best_estimator_


# In[32]:


max(scores['mean_train_score'])


# In[33]:


XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.0, learning_rate=0.1,
       max_delta_step=0, max_depth=3, min_child_weight=5, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)


# In[34]:


XGB.fit(X_train, y_train)


# In[35]:


XGB.score(X_test,y_test)


# In[36]:


Grid1 = GridSearchCV(RF, parameters2, cv=3,return_train_score=True)

Grid1.fit(X_train,y_train)


# In[37]:


scores = Grid1.cv_results_


# In[38]:


Grid1.best_estimator_


# In[39]:


max(scores['mean_train_score'])


# In[40]:


pred = XGB.predict(test)


# In[41]:


result = pd.DataFrame(pred,columns=['Survived'])


# In[42]:


test  = pd.read_csv('test.csv' , header = 0, dtype={'Age': np.float64})


# In[43]:


submission = result.join(test['PassengerId']).iloc[:,::-1]


# In[44]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




