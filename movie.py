#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
from plotly.offline import iplot
from plotly.subplots import make_subplots
from sklearn.datasets import load_boston

import warnings 
warnings.simplefilter(action="ignore")
sns.set_theme(palette=sns.color_palette("muted"),style="darkgrid")


# In[2]:


pip install --upgrade xgboost


# In[3]:


pip install lightgbm


# In[4]:


pip install category-encoders


# In[5]:


pip install scikit-plot


# In[6]:



pip install xgboost


# In[7]:



pip install sklearn


# In[8]:


data=pd.read_csv("Movies.csv")


# In[9]:


data.sample(10)


# In[10]:


data.info()


# In[11]:


data.isna().sum()


# In[12]:


data=data.dropna(subset=["Year","Genre","Director","Actor 1","Actor 2","Actor 3","Rating"])


# In[13]:


data=data.dropna().reset_index(drop=True)


# In[14]:


data.shape


# In[15]:


data["Duration"]=data.loc[:,"Duration"].str.replace(" min","")
data["Duration"] = pd.to_numeric(data.loc[:,"Duration"])


# In[16]:


data["Votes"]=data.loc[:,"Votes"].str.replace(",","")
data["Votes"] =pd.to_numeric(data.loc[:,"Votes"])


# In[17]:


data.shape


# In[18]:


data.duplicated(subset=["Name","Year"]).sum()


# In[19]:


data=data.drop_duplicates(keep="first",subset=["Name","Year"]).reset_index(drop=True)


# In[20]:


data.columns


# In[21]:


data["Year"] = data["Year"].astype(str)
data["Year"] = data.loc[:,"Year"].str.extract(r"(\d{4})")
data["Year"] = pd.to_numeric(data.loc[:,"Year"])


# In[22]:


data["Genre"]=data.loc[:,"Genre"].str.extract("(^\w{1,11})")
data["Main_genre"]=data.loc[:,"Genre"].str.replace("Musical","Music")


# In[23]:


data["Main_genre"].unique()


# In[24]:


data.describe()


# In[25]:


data=data[(np.abs(stats.zscore(data[['Rating','Votes','Duration']]))<3).all(axis=1)]


# In[26]:


data.shape


# In[27]:


iplot(px.violin(data_frame=data,x="Rating"))


# In[28]:


iplot(px.violin(data_frame=data,x="Duration"))


# In[29]:


iplot(px.violin(data_frame=data,x="Votes"))


# In[30]:


data.Genre.value_counts().reset_index()


# In[31]:


genres=data.Main_genre.value_counts().reset_index()
genres.columns = ['Main_genre', 'count']
iplot(px.pie(data_frame=genres,names=genres.Main_genre,values=genres["count"],title="Number Of Movies by Genre",height=1050).update_traces(textinfo="value+percent"))


# In[32]:


data.groupby("Main_genre")["Rating"].mean().sort_values(ascending=False)


# In[33]:


rating_by_genre=data.groupby("Main_genre")["Rating"].mean().sort_values(ascending=False)
iplot(px.bar(data_frame=rating_by_genre))


# In[34]:


movies_by_year = data['Year'].value_counts().reset_index()
movies_by_year.columns = ['Year', 'count']
movies_by_year = movies_by_year.sort_values(by='Year')
iplot(px.line(data_frame=movies_by_year,x="Year",y="count",title="Number of Movies over the years",color_discrete_sequence=["green"]))


# In[35]:


Rating_by_years =data.groupby("Year").agg({"Rating":"mean","Votes":"sum"}).reset_index()


# In[36]:


iplot(px.line(data_frame=Rating_by_years,x="Year",y="Rating",markers=True,color_discrete_sequence=["green"],height=400))


# In[37]:


iplot(px.line(data_frame=Rating_by_years,x="Year",y="Votes",color_discrete_sequence=["Red"],markers=True,height=400))


# In[38]:


def top_10_rating(col):
    return data.groupby(col)["Rating"].agg(["mean","count"]).query("count>=10").sort_values(by="mean",ascending=False)[:10].reset_index() 


# In[39]:


top_10_director = top_10_rating("Director")
iplot(px.bar(data_frame=top_10_director ,x="Director",y="mean",text="count",labels={'mean':'Rating','count':'Number of movies'},title="Top 10 Directors with more than 10 movies by rating"))


# In[40]:


top_10_actors1=top_10_rating("Actor 1")
iplot(px.bar(data_frame=top_10_actors1,x="Actor 1",y="mean",text="count",labels={'Actor 1':'Main Actor','mean':'Rating','count':'Number of movies'},title="Top 10 Main actors with more than 10 movies by rating"))


# In[41]:


top_10_actors2=top_10_rating("Actor 2")
iplot(px.bar(data_frame=top_10_actors2,x="Actor 2",y="mean",text="count",labels={'mean':'Rating','count':'Number of movies'},title="top 10 secondary actors with more than 10 movies by rating"))


# In[42]:


top_10_actors3=top_10_rating("Actor 3")
iplot(px.bar(data_frame=top_10_actors3,x="Actor 3",y="mean",text="count",labels={'mean':'Rating','count':'Number of movies'},title="Top 10 Third main actors with more than 10 movies by rating"))


# In[43]:


from sklearn.model_selection import train_test_split,cross_val_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.preprocessing import RobustScaler,StandardScaler,MaxAbsScaler,MinMaxScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
import sklearn.metrics as metrics
import scikitplot as skplt
import category_encoders as ce


# In[44]:


def regression_results(y_true,y_pred):
    explained_variance=metrics.explained_variance_score(y_true,y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true,y_pred)
    mse = metrics.mean_squared_error(y_true,y_pred)
    mean_squared_log_error=metrics.mean_squared_log_error(y_true,y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true,y_pred)
    r2=metrics.r2_score(y_true,y_pred)
    print('explained_variance :',round(explained_variance,4))
    print('mean_squared_log_error:',round(mean_squared_log_error,4))
    print('r2:' ,round(r2,4))
    print('MAE:',round(mean_absolute_error,4))
    print('MSE:',round(mse,4))
    print('RMSE :',round(np.sqrt(mse),4))
    print('Median absolute error :' ,round(median_absolute_error,4))


# In[45]:


data=data.drop(columns=["Name","Main_genre"])


# In[46]:


X=data.drop(columns="Rating")
y=data["Rating"]


# In[47]:


X


# In[48]:


y


# In[49]:


encoder =ce.JamesSteinEncoder(return_df=True)


# In[50]:


encoder.fit(X,y)
X = encoder.transform(X.loc[:,:])


# In[51]:


X


# In[52]:


scaler =RobustScaler()
scaler.fit(X)
X.loc[:,:]=scaler.transform(X.loc[:,:])


# In[53]:


X.columns


# In[54]:


import xgboost as xgb
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=7,shuffle=True)


# In[55]:


xgb_model = XGBRegressor(objective ='reg:squarederror',gamma=0.09,learning_rate=0.08,subsample=0.7)


# In[56]:


xgb_model.fit(X_train,y_train)


# In[57]:


xgb_model.score(X_train,y_train)


# In[58]:


xgb_model.score(X_test,y_test)


# In[59]:


y_pred = xgb_model.predict(X_test)
print(f"Report:Lgbm model")
print(f"{regression_results(y_test, y_pred)}\n")


# In[60]:


score = cross_val_score(xgb_model,X,y,cv=10)
avg = np.mean(score)
print(f"cross validation score for XGBoost:{score}")
print(f"average cross validation score for XGBoost:{avg}\n")


# In[61]:


fs = xgb_model.feature_importances_
feature_names = X.columns

feature_importances = pd.DataFrame(fs,feature_names).sort_values(by=0,ascending=False)
plt.figure(figsize=(12, 9))
plt.title("Feature Importances")
plt.bar(x=feature_importances.index,height=feature_importances[0])
plt.xticks(rotation=90)

plt.show()


# In[62]:


data.head(15)


# In[63]:


new_data = pd.DataFrame({'Year':[2015],'Duration':[115],                         'Genre':'Comedy,Drama','Votes':[7123],'Director':['Sharat Katariya'],                         'Actor 1':['Jeet'],'Actor 2':['Rishi Kapoor'],'Actor 3':['Vijay Raaz'],})


# In[64]:


new_data


# In[65]:


new_data = encoder.transform(new_data)
new_data.loc[:,:] = scaler.transform(new_data.loc[:,:])


# In[66]:


new_data


# In[67]:


xgb_model.predict(new_data)

