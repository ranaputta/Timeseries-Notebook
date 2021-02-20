#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# 1. Load the CSV data into a pandas data frame. Print some high-level statistical info about the data frame's columns.

# In[7]:


df=pd.read_csv("Concrete_Data.csv")
df.head()


# 2. How many rows have a compressive strength > 40 MPa?

# In[8]:


sum(df['Concrete_Compressive_Strength']>40)


# 3. Plot the histogram of Coarse Aggregate and Fine Aggregate values

# In[9]:


df.hist("Coarse_Aggregate")


# In[10]:


df.hist("Fine_Aggregate")


# 4. Make a plot comparing compressive strength to age

# In[11]:


df.plot(x="Age",y="Concrete_Compressive_Strength",kind="scatter")


# 5. Make a plot comparing compressive strength to age for only those rows with < 750 fine aggregate.

# In[12]:


df2=df[df["Fine_Aggregate"]<750]


# In[13]:


df2.head()


# In[14]:


df2.plot(x="Age",y="Concrete_Compressive_Strength",kind="scatter")


# 6. Try to build a linear model that predicts compressive strength given the other available fields.

# In[23]:


from sklearn import linear_model 

lin_m = linear_model.Lasso(alpha=0.01)

y=df["Concrete_Compressive_Strength"]
x=df.drop("Concrete_Compressive_Strength",axis=1)

lin_m.fit(x,y)

pd.DataFrame([dict(zip(x, lin_m.coef_))])


# 7. Generate predictions for all the observations and a scatterplot comparing the predicted compressive strengths to the actual values.

# In[30]:


predictm = lin_m.predict(x)
predict_df = df.assign(prediction=predictm)
predict_df[["Concrete_Compressive_Strength", "prediction"]]


# In[32]:


predict_df.plot(kind="scatter", x="Concrete_Compressive_Strength", y="prediction")


# In[ ]:




