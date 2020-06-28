#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv("titanic.csv")
df.head()


# In[4]:


df.info()


# In[5]:


df.nunique()


# In[6]:


df.isnull().sum().sum()


# In[7]:


df.isnull().sum()


# In[8]:


df.drop(labels = ["Cabin", "PassengerId", "Embarked", "Ticket"], axis=1).head()


# In[9]:


df['Age'].fillna(df['Age'].mean())
# The preprocession part is now finished.


# In[10]:


df["Age"].isnull().sum()
# Simple check


# In[11]:


df.groupby(["Sex"]).mean()
#the pourcentage of Females that has survived is much more important for Females than for Males


# In[12]:


df.groupby(["Survived"]).mean()


# In[13]:


df.groupby(["Pclass","Sex"]).mean()


# In[14]:


def plot_correlation_map( df ):

    corr = df.corr()

    s , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    s = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

        )


# In[15]:


plot_correlation_map( df )
# Ici plus la valeur absolue de la valeur indiquée par la grille se rapproche de 1 plus il y a corrélation entre les 2 paramères reliés. Par exemple la case P/class et survived est à 0.35.Cela veut dire qu'il y a une grande relation entre la survie d'un individu et sa classe de de départ. A contrario la survie d'un individu n'a pas pour paramètre son PassengerId comme nous l'indique la case Survived/PasssengerId à 0.005


# In[16]:


df[["Pclass", "Survived"]].groupby(["Pclass"], as_index = True).mean()


# In[17]:


grid = sns.FacetGrid(df,col='Survived',
    row='Sex', height=2.2, aspect=1.6)
grid.map(plt.hist,'Age', alpha=.5, bins=20)
grid.add_legend()


# In[18]:


g = sns.FacetGrid(df,col='Survived',
    row='Pclass', height=2.2, aspect=1.6)
g.map(plt.hist,'SibSp', alpha=.5, bins=20)
g.add_legend()


# In[19]:


Title_Dictionary = {

                    "Capt":       "Officer",

                    "Col":        "Officer",

                    "Major":      "Officer",

                      "Dr":         "Officer",

                    "Rev":        "Officer",

                    "Jonkheer":   "Royalty",

                    "Don":        "Royalty",

                    "Sir" :       "Royalty",

                   "Lady" :      "Royalty",

                  "the Countess": "Royalty",

                    "Dona":       "Royalty",

                    "Mme":        "Miss",

                    "Mlle":       "Miss",

                    "Miss" :      "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mrs",

                    "Mrs" :       "Mrs",

                    "Master" :    "Master"}


# In[20]:


title=[]
df1=' '
for data in df['Name']:
    
    df1=data.split(',') 
    
    df1=df1[1].split('.')
    df2=df1[0].strip()
    print(df2)
    title.append(str(df2))
df['title']=title 
df.head()


# In[21]:


df['title']=df['title'].map(Title_Dictionary)


# In[22]:


df.head()


# In[23]:


df['title'].values


# In[24]:


df["title"].value_counts()


# In[25]:


# Let's visualize the correlation between title and other features.
df.groupby(["title"]).mean()


# In[26]:


df[["title", "Survived"]].groupby(["title"], as_index = True).mean()


# In[27]:


# We create a new unattached column. It's data consist on the addition of the value corresponding to the Parch feature and the value corresponding to the SibSp feature on each row, thanks to the lambda function below.
df.apply(lambda row: row.Parch + row.SibSp, axis = 1)
# We attach this new column to our dataframe.
df["FamilySize"] = df.apply(lambda row: row.Parch + row.SibSp, axis = 1)
# Once we have created the FamilySize column, we drop the two initial less useful columns : Parch and SibSp.
df = df.drop(columns = ["Parch", "SibSp"])
df.head()


# In[ ]:




