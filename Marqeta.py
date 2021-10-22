#!/usr/bin/env python
# coding: utf-8

# ### Import required libraries

# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import norm
from statsmodels.tsa.seasonal import seasonal_decompose


# ### Read csv files as dataframes

# In[51]:


df_counts = pd.read_csv('counts.csv.gz')


# In[52]:


df_spend = pd.read_csv('spend.csv.gz')


# In[53]:


df_counts['date'] = pd.to_datetime(df_counts['date']).dt.date


# In[54]:


df_counts.info()


# In[55]:


df_spend['date'] = pd.to_datetime(df_spend['date'],unit='s').dt.date


# ### Analyzing for missing or incorrect data

# In[56]:


df_spend.isna().sum() ##no missing values in spend df


# In[57]:


df_counts.isna().sum() ##no missing values in counts df


# In[58]:


##creating random date column to analyze any outlier in 'date' field
df_spend['rand_date'] = pd.to_datetime('2021-10-15')
df_counts['rand_date'] = pd.to_datetime('2021-10-15')


# In[59]:


df_spend['rand_date'] = df_spend['rand_date'].dt.date
df_counts['rand_date'] = df_counts['rand_date'].dt.date


# In[60]:


##taking a difference with the 'date' field for quick box-plot analysis
df_spend['days_diff'] = df_spend['rand_date'] - df_spend['date']
df_counts['days_diff'] = df_counts['rand_date'] - df_counts['date']


# In[192]:


df_spend.head()


# In[193]:


df_counts.head()


# In[63]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[173]:


plt.boxplot(df_spend.days_diff) ##identified outlier in 'date' in spend df


# In[65]:


plt.boxplot(df_counts.days_diff) ##no outlier in 'date' field in counts df


# In[185]:


## running 5 number statistics to get outlier values

# 1st quartile
q1 = np.quantile(df_spend['days_diff'], 0.25)
 
# 3rd quartile
q3 = np.quantile(df_spend['days_diff'], 0.75)
 
# Inter-quartile range
iqr = q3-q1

# Upper and Lower whiskers
upper_bound = q3+(1.5*iqr)
lower_bound = q1-(1.5*iqr)


# In[188]:


outliers = df_spend['days_diff'][(df_spend['days_diff'] <= lower_bound) | (df_spend['days_diff'] >= upper_bound)]
outliers


# In[69]:


df_spend_incorrect_data = df_spend.loc[df_spend['days_diff'] == '17725 days']
df_spend_incorrect_data


# In[189]:


df_counts.loc[df_counts['account']=='a04e063493c55bc6cecd9056712e9e47']

df_incorrect_data_join = pd.merge(df_counts.loc[df_counts['account']=='a04e063493c55bc6cecd9056712e9e47'], df_spend.loc[df_spend['account']=='a04e063493c55bc6cecd9056712e9e47'], on = ['account','date'], how='outer', indicator=True)


# In[190]:


df_incorrect_data_join.loc[df_incorrect_data_join['_merge']!='both']


# In[196]:


##replacing the outlier date value in spend df with the only non-matching record from counts df

df_spend.loc[1,'date'] = ['2017-08-16']


# ### Aggregating amount and volume of transactions per account

# In[71]:


df_counts_agg = df_counts.groupby(['account'], as_index=False)['count'].sum()


# In[201]:


df_counts_agg.head()


# In[72]:


df_counts_agg.describe()


# In[200]:


df_spend_agg = df_spend.groupby(['account'], as_index=False)['amount'].sum()


# In[202]:


df_spend_agg.head()


# In[228]:


##plot distribution of spend aggregate for accounts
x_axis = np.arange(0, 2000000, 0.01)
  
# Calculating mean and standard deviation
mean = df_spend_agg.mean()
std = df_spend_agg.std()
  
plt.plot(x_axis, norm.pdf(x_axis, mean, std))
plt.show()


# In[227]:


##plot distribution of counts aggregate for accounts
x_axis = np.arange(0, 200000, 0.01)
  
# Calculating mean and standard deviation
mean = df_counts_agg.mean()
std = df_counts_agg.std()
  
plt.plot(x_axis, norm.pdf(x_axis, mean, std))
plt.show()


# In[74]:


df_spend_agg.describe()


# In[75]:


df_counts_agg.info()


# In[76]:


sns.boxplot(df_counts_agg['count'])


# In[77]:


df_counts_agg.loc[df_counts_agg['count']>700000]


# In[209]:


# 1st quartile
q1 = np.quantile(df_counts_agg['count'], 0.25)
 
# 3rd quartile
q3 = np.quantile(df_counts_agg['count'], 0.75)
 
# Inter-quartile range
iqr = q3-q1

# Upper and Lower whiskers
upper_fence = q3+(1.5*iqr)
lower_fence = q1-(1.5*iqr)

# Extreme Upper and Lower fence
e_upper_fence = q3+(3*iqr)
e_lower_fence = q1-(3*iqr)


# In[211]:


e_upper_fence


# In[213]:


df_counts_agg.loc[df_counts_agg['count']>e_upper_fence]


# In[78]:


sns.boxplot(df_spend_agg['amount'])


# In[79]:


df_spend_agg.loc[df_spend_agg['amount']>3000000]


# In[80]:


df_agg_join = pd.merge(df_spend_agg, df_counts_agg, on = 'account', how='left', indicator=True)


# In[81]:


df_agg_join.loc[df_agg_join['_merge']!='both']


# In[82]:


df_agg_join.head()


# In[83]:


df_agg_join['avg_trans_amt']=df_agg_join['amount']/df_agg_join['count']


# In[84]:


df_agg_join.head()


# In[85]:


df_agg_join['avg_trans_amt'].describe()


# In[144]:


sns.boxplot(df_agg_join['avg_trans_amt'])


# In[155]:


# 1st quartile
q1 = np.quantile(df_agg_join['avg_trans_amt'].loc[df_agg_join['avg_trans_amt']<10000], 0.25)
 
# 3rd quartile
q3 = np.quantile(df_agg_join['avg_trans_amt'].loc[df_agg_join['avg_trans_amt']<10000], 0.75)
 
# Inter-quartile range
iqr = q3-q1

# Upper and Lower whiskers
upper_fence = q3+(1.5*iqr)
lower_fence = q1-(1.5*iqr)

# Extreme Upper and Lower fence
e_upper_fence = q3+(3*iqr)
e_lower_fence = q1-(3*iqr)


# In[164]:


e_lower_fence, lower_fence, q1, df_agg_join['avg_trans_amt'].median(), q3, upper_fence, e_upper_fence


# In[158]:


mild_outliers = df_agg_join['avg_trans_amt'][(df_agg_join['avg_trans_amt'] <= lower_fence) | (df_agg_join['avg_trans_amt'] >= upper_fence)]
extreme_outliers = df_agg_join['avg_trans_amt'][(df_agg_join['avg_trans_amt'] <= e_lower_fence) | (df_agg_join['avg_trans_amt'] >= e_upper_fence)]


# In[167]:


mild_outliers


# In[168]:


extreme_outliers


# In[161]:


sns.boxplot(df_agg_join['avg_trans_amt'].loc[(df_agg_join['avg_trans_amt'] > lower_fence) & (df_agg_join['avg_trans_amt'] < upper_fence)])


# In[162]:


sns.boxplot(df_agg_join['avg_trans_amt'].loc[(df_agg_join['avg_trans_amt'] > e_lower_fence) & (df_agg_join['avg_trans_amt'] < e_upper_fence)])


# In[248]:


plot = df_agg_join['avg_trans_amt'].loc[(df_agg_join['avg_trans_amt'] > lower_fence) & (df_agg_join['avg_trans_amt'] < upper_fence)].plot()
plot.axhline(df_agg_join['avg_trans_amt'].mean(),color='red')


# In[245]:


##plot distribution of counts aggregate for accounts
x_axis = np.arange(-500, 5500, 0.01)
  
# Calculating mean and standard deviation
mean = df_agg_join['avg_trans_amt'].mean()
std = df_agg_join['avg_trans_amt'].std()
  
plt.plot(x_axis, norm.pdf(x_axis, mean, std))
plt.show()


# In[242]:


df_agg_join['avg_trans_amt'].min()


# In[40]:


df_agg_join.head()


# In[41]:


df_counts.head()


# ### Time-series decomposition

# In[42]:


df_counts_trans = df_counts.groupby(['date'], as_index=False)['count'].sum()


# In[43]:


df_counts_trans.head()


# In[44]:


df_counts_trans['date']=pd.to_datetime(df_counts_trans['date'])


# In[45]:


df_counts_trans.set_index('date',inplace=True)
df_counts_trans.index=pd.to_datetime(df_counts_trans.index)

df_counts_trans.plot()


# In[265]:


result=seasonal_decompose(df_counts_trans['count'], model='additive')


# In[266]:


result.seasonal.plot()


# In[267]:


result.trend.plot()


# In[268]:


result.plot()


# In[ ]:




