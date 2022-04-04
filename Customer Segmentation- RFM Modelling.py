#!/usr/bin/env python
# coding: utf-8

# ### As a Data Scientist | customer Analyst, my interest is the development of predictive segments using the RFM modelling to help Marketing Managers better serve their customers and maximize their customer lifetime value. 
# 
# #### The RFM (Recency, Frequency, and Monetary) model categorises customers based on their transaction history - how recently, how often and what their transaction is worth.
# 
# 

# ### The RFM Analysis answers business questions like:
# 
# #### Who are the best customers?
# 
# #### Which customers are on the verge of churning?
# 
# #### who are lost customers that the business can afford to ignore to effectively utilize budgets? 
# 
# #### Which customer is likely to be loyal in the near future?

# #### This model is linked to the popular 80/20 pareto principle which when applied to marketing indicates that 80% of the total revenue likely comes from the top 20% of the customers, thus making the identification and retention of such customers highly critical for business success.

# ## Methodology
# #### This analysis leverages a historical retail transaction data which contains customers' transactions in a given time period. Customized filters would be created to effectively segment the customers. Python would be used to obtain the frequency, recency and monetary values in the last 365 days for each customer, so that they can be grouped into different segments.

# In[1]:


#importing the libraries
import pandas as pd
import warnings
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')


# In[2]:


#Loading the dataset
rfmdata= pd.read_csv('Retail_Data_Transactions.csv')
rfmdata.head()


# In[3]:


# This shows that the dataset contains 3 columns and 124999 rows of observation
rfmdata.info()


# In[4]:


#Checking the details of customers transactions.
rfmdata.describe()


# #### This reveals that most of the customers per transaction, spend £105 on the avearage with £10 being the lowest and £105 being the hightest value of spend.  
# 

# In[5]:


#Total number of customers 
rfmdata['customer_id'].nunique()


# In[8]:


# change the tran-date column to datetime
rfmdata['trans_date'] = pd.to_datetime(rfmdata['trans_date'])
rfmdata.info()


# In[9]:


rfmdata.head()


# #### Checking for the earliest and lastest transaction date

# In[10]:


rfmdata['trans_date'].min()


# In[11]:


rfmdata['trans_date'].max()


# In[12]:


#Assuming this analysis is being done on the 01/04/2015,finding the number of days from the
#last transaction date which would be used to calculate the recency value.
sd = dt.datetime(2015,4,1)
rfmdata['hist']=sd - rfmdata['trans_date']
rfmdata['hist'].astype('timedelta64[D]')
rfmdata['hist']=rfmdata['hist'] / np.timedelta64(1, 'D')
rfmdata.head()


# In[14]:


# considering only transactions done within the ast 2 years (730 days)
rfmdata=rfmdata[rfmdata['hist'] < 730]
rfmdata.info()


# ### Visualizing the Recency Value

# In[36]:


plt.figure(figsize=(8,5))
sns.distplot(rfmSeg.recency,bins=8,kde=False,rug=True)

#The visualization shows that the number of customers that transacted with the business within the last 90 days are more than those that didn't within the last two years. 


# ### Visualizing the Frequency Value

# In[37]:


plt.figure(figsize=(8,5))
sns.distplot(rfmSeg.frequency,bins=8,kde=False,rug=True)
#The plot shows that the number of customers that transacted between 7 tp 9 times is far more than those that did 5 times or 15 times. 


# ### Visualizing the Monetary Value

# In[62]:


plt.figure(figsize=(8,5))
sns.distplot(rfmSeg.monetary,bins=8,kde=False,rug=True)
#The monetary value is the sum of the value of each customer's transaction.
#The plot shows that most customers spent in total between £400 and £600 with the last two years.  


# #### Visualizing Recency VS Frequency

# In[55]:


plt.scatter(rfmSeg.groupby('customer_id')['recency'].sum(), rfmSeg.groupby('customer_id')['frequency'].sum(),
            color = 'red',
            marker = '*', alpha = 0.3)

plt.title('Scatter Plot for Recency and Frequency') 
plt.xlabel('recency')
plt.ylabel('frequency')

#The plot shows that the customers that bought most recently, transact most frequently with the business.


# #### visualizing Frequency vs Monetary
#              

# In[56]:



plt.scatter(rfmSeg.groupby('customer_id')['monetary'].sum(), rfmSeg.groupby('customer_id')['frequency'].sum(),
            color = 'red',
            marker = '*', alpha = 0.3)

plt.title('Scatter Plot for Monetary and Frequency') 
plt.xlabel('Monetary')
plt.ylabel('Frequency')

#The plot is showing that the highest spender buy most frequently and there is a concentration of customers around the average for both frequency and monetary.


# #### Recency VS Frequency VS Monetary

# In[48]:


Monetary = rfmdata.groupby('customer_id')['tran_amount'].sum()
plt.scatter(rfmSeg.groupby('customer_id')['recency'].sum(), rfmSeg.groupby('customer_id')['frequency'].sum(),
            marker = '*', alpha = 0.3,c= Monetary)

plt.title('Scatter Plot for monetary, Recency and Frequency')
plt.xlabel('recency')
plt.ylabel('Frequency')

#The colours indicate monetary value. The customers with higher spending power have lower frequency with varying recency values.


# ### Creating and validating the RFM Table

# In[15]:


rfmTable = rfmdata.groupby('customer_id').agg({'hist': lambda x:x.min(), # Recency
                                        'customer_id': lambda x: len(x), # Frequency
                                        'tran_amount': lambda x: x.sum()})# Monetary

rfmTable.rename(columns={'hist': 'recency', 
                         'customer_id': 'frequency', 
                         'tran_amount': 'monetary'}, inplace=True)

rfmTable.head()


# #### The RFM table shows that customer CS1114 has the highest monetary value of £804, purchased 11 times and his last transaction day was 48 days ago

# In[16]:


rfmdata[rfmdata['customer_id']=='CS1114']


# #### To validate the RFM table, the transaction history of customer CS1114 was used for cross-checking and the result proved the RFM table to be accurate.

# #### The quintile method applied to obtain the the RFM Scores.

# In[17]:


quintiles = rfmTable.quantile(q=[0.20,0.40,0.60,0.80])
quintiles


# #### Inserting quintles into a dictionary for ease of use
# 

# In[18]:


quintiles=quintiles.to_dict()
quintiles


# In[19]:


rfmSeg = rfmTable


# #### Scores between 1 and 5 are being assigned to Recency, Frequency and Monetary, 5 being the best score and 1 being the least.

# In[31]:


# Arguments (x = value, p = recency, monetary_value, frequency, k = quintile dict)
#Finding the Recency, freqency and Monetary for each customer

def RFMClass(x,p,d):
    if x <= d[p][0.20]:
        return 5
    elif x <= d[p][0.40]:
        return 4
    elif x <= d[p][0.60]: 
        return 3
    elif x <= d[p][0.80]: 
        return 2
    else:
        return 1


# In[32]:


rfmSeg = rfmTable
rfmSeg['R_score'] = rfmSeg['recency'].apply(RFMClass, args=('recency',quintiles,))
rfmSeg['F_score'] = rfmSeg['frequency'].apply(RFMClass, args=('frequency',quintiles,))
rfmSeg['M_score'] = rfmSeg['monetary'].apply(RFMClass, args=('monetary',quintiles,))


# In[33]:


#Creating the RFM score by concartinating all scores 

rfmSeg['RFMscore'] = rfmSeg.R_score.map(str)                             + rfmSeg.F_score.map(str)                             + rfmSeg.M_score.map(str)


# In[34]:


rfmSeg.head()


# #### Every customer has now been assigned their RFM score, showing what category they belong to.

# In[35]:


#Sorting the RFM in asscending order
rfmSeg.sort_values(by=['RFMscore', 'monetary'], ascending=[True, False])


# In[60]:


rfmSeg[rfmSeg['RFMscore']=='555'].sort_values('monetary', ascending=False).head(5)


# #### The values above show the top 5 customers with RFM score- 555, that interact with the business frequently, not long ago, then are the biggest spenders. 

# ### The insights provided by this RFM modelling would help this business engage with its customer groups differently with targeted marketing strategies.

# In[ ]:




