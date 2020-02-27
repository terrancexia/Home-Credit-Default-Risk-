#!/usr/bin/env python
# coding: utf-8

# # Home Credit Default Risk - Completed

# ## Background Information
# 
# Home Credit is an international finance provider funded in 1997 in Czech Republic with operation in 9 different countries. The task is to used dataset provided and predict whether or nor a client will repay a loan or have difficulty repay the loan. 
# 

# ## Data Description
# 
# Data is provided by Home Credit. Data can be download at Kaggle open dataset: https://www.kaggle.com/c/home-credit-default-risk/data
# 
# The Dataset including the following source:
# 
# 1. application_train/application_test: the main training and testing data with information about each loan application at Home Credit. Every loan has its own row and is identified by the feature SK_ID_CURR. The training application data comes with the TARGET indicating 0: the loan was repaid or 1: the loan was not repaid. 
#     
#     
# 2. bureau.csv: All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample). For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.
# 
# 
# 3. bureau_balance.csv: monthly balances of previous credits in Credit Bureau.This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.
# 
# 
# 4. POS_CASH_balance.csv: Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit. This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.
# 
# 
# 5. credit_card_balance.csv: Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.
# 
# 
# 6. previous_application.csv:All previous applications for Home Credit loans of clients who have loans in our sample.There is one row for each previous application related to loans in our data sample.
# 
# 
# 7. installments_payments.csv: Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample. There is a) one row for every payment that was made plus b) one row each for missed payment.One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.
# 
# 
# 8. HomeCredit_columns_description.csv:This file contains descriptions for the columns in the various data files.
# 
# 
# ![home_credit.png](image/home_credit.png)

# # Import Packages and Data

# In[1]:


#import packages
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats
import statsmodels.api as sm 

pd.set_option('display.max_columns',None)

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# import data 
app_train = pd.read_csv("Data/application_train.csv")
app_test =pd.read_csv("Data/application_test.csv")
bureau = pd.read_csv("Data/bureau.csv")
bur_bal = pd.read_csv("Data/bureau_balance.csv")
pos_cash =pd.read_csv("Data/POS_CASH_balance.csv")
cc_balance = pd.read_csv("Data/credit_card_balance.csv")
py_bal = pd.read_csv("Data/previous_application.csv")
ins_pmt= pd.read_csv("Data/installments_payments.csv")


# In[3]:


#inspect data
app_train.head()


# In[4]:


app_test.head()


# In[5]:


bureau.head() 


# In[6]:


bur_bal.head() 


# In[7]:


pos_cash.head()


# In[8]:


cc_balance.head() 


# In[9]:


py_bal.head() 


# In[10]:


ins_pmt.head()


# # Data Cleaning

# ## Combining training and testing
# 
# Before we cleaning the data we need to combine training and testing so that the data cleaning process can apply to both data set for consistent

# In[11]:


# Checking whether training and testing have same columns
app_train.columns.isin(app_test.columns)


# In[12]:


print(app_train.columns[1])

print(f'Number of columns in training set', len(app_train.columns))
print(f'Number of columns in testing set', len(app_test.columns))


# Looks like the columns in training and testing are the same except the target column. Therefore, I will add a 'Target' column in the testing set

# In[13]:


#add empty column Target into the app_test
app_test['TARGET']=np.nan

#sanity check
print(app_train.columns.isin(app_test.columns))

print(f'Number of columns in training set', len(app_train.columns))
print(f'Number of columns in testing set', len(app_test.columns))


# In[14]:


#Combining dataset

df = pd.concat([app_train,app_test])

#sanity check
print(f'Shape of App_train:', app_train.shape)
print(f'Shape of app_test:', app_test.shape)
print(f'Shape of df:', df.shape)


# In[15]:


# Check df info
df.info(max_cols=122)


# ## Missing data

# In[16]:


#check on the missing data for app_train
total_miss = df.isnull().sum().sort_values(ascending=False)
miss_percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
miss_summ = pd.concat([total_miss, miss_percent], axis=1, keys=['Total Missing','Missing Percent'])
miss_summ


# ### Current living area measurement
# Based on above missing data summary app_train, we can see that most of the missing columns involved missing information about their current living area measurement. People failed to put this information maybe simply due to they do not know what are the exact measurement of their current living places, especially to each floor, living area, non-living area.  For the purpose of this prediction, I assume the current detailed living area has no impact on whether or not the client can repay the loan. Therefore, I will delete those columns related to detailed measurement of current living places.  

# In[17]:


# check the index names related to current living area detailed measurement in miss_summ. Based on the miss_summ 
# all of the measurement are missing more than 40%
miss_summ[miss_summ['Missing Percent']>40].index


# In[18]:


# Delete the columns related to current living area detailed measurement. 
df1 = df.drop(['COMMONAREA_MEDI', 'COMMONAREA_AVG', 'COMMONAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 
                             'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAPARTMENTS_AVG', 'FONDKAPREMONT_MODE', 
                             'LIVINGAPARTMENTS_MEDI', 'LIVINGAPARTMENTS_MODE', 'LIVINGAPARTMENTS_AVG', 
                             'FLOORSMIN_MEDI', 'FLOORSMIN_MODE', 'FLOORSMIN_AVG', 'YEARS_BUILD_MEDI', 
                             'YEARS_BUILD_AVG', 'YEARS_BUILD_MODE', 'OWN_CAR_AGE', 'LANDAREA_MODE', 'LANDAREA_AVG', 
                             'LANDAREA_MEDI', 'BASEMENTAREA_MEDI', 'BASEMENTAREA_AVG', 'BASEMENTAREA_MODE', 
                             'NONLIVINGAREA_MEDI', 'NONLIVINGAREA_AVG', 'NONLIVINGAREA_MODE', 'ELEVATORS_MODE', 
                             'ELEVATORS_AVG', 'ELEVATORS_MEDI', 'WALLSMATERIAL_MODE', 'APARTMENTS_MODE', 'APARTMENTS_AVG', 
                             'APARTMENTS_MEDI', 'ENTRANCES_MEDI', 'ENTRANCES_MODE', 'ENTRANCES_AVG', 'LIVINGAREA_MEDI', 
                             'LIVINGAREA_MODE', 'LIVINGAREA_AVG', 'HOUSETYPE_MODE', 'FLOORSMAX_MODE', 'FLOORSMAX_MEDI', 
                             'FLOORSMAX_AVG', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BEGINEXPLUATATION_AVG', 
                             'YEARS_BEGINEXPLUATATION_MODE', 'TOTALAREA_MODE','EMERGENCYSTATE_MODE'], axis=1)


# In[19]:


#recheck the missing data summary
total_miss1 = df1.isnull().sum().sort_values(ascending=False)
miss_percent1 = (df1.isnull().sum()/df1.isnull().count()*100).sort_values(ascending=False)
miss_summ1 = pd.concat([total_miss1, miss_percent1], axis=1, keys=['Total Missing','Missing Percent'])
miss_summ1[miss_summ1['Missing Percent']>0]


# ### Ext_source_1, 2 & 3
# 
# Ext_Source_1, 2 and 3 are the normalized scores from external source. Most likely these three scores are some kind of credit report for the applicant. Since they are the external measurement of the applicant, they are crucial parts of the data. Therefore, we will examine the data and fill in the missing value. Based on the histograms below, it seems the Ext_source_1 is normally distributed, while Ext_source_2 and Ext_source_3 are skewed to the left. The distribution of Ext_source_2 and Ext_source_3 are similar and they are different than Ext_source_1. 
# 

# In[20]:


#Check the summary stats for ext_source_1
df1[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].describe()


# In[21]:


#Plot out the distribution for EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3
plt.figure(figsize =(10,5))
plt.hist(df1['EXT_SOURCE_1'], edgecolor='black', bins = 100)
plt.title("Histogram of Ext_source_1")

plt.figure(figsize=(10,5))
plt.hist(df1['EXT_SOURCE_2'], edgecolor='black', bins = 100)
plt.title("Histogram of Ext_source_2")

plt.figure(figsize=(10,5))
plt.hist(df1['EXT_SOURCE_3'], edgecolor='black', bins = 100)
plt.title("Histogram of Ext_source_3")


plt.show()


# For missing data in the ext_source_1, 2, and 3, it would be a good idea to use regression to impute the missing data. However, we need to test whether there will be any correlation between each of the external source and the correlation of each external source vs all other columns. However, based on the correlation test below, each of the external source has no strong statistical correlation with each other. Also each of the external source has no strong statistical correlation with all other columns. Using regression to impute missing data will impose bias on the predicted values. Therefore, I will impute the missing values of Ext_source_1 with mean and missing values of Ext_source_2, Ext_source_3 with the median of each columns.   

# In[22]:


#Check on the correlation of the three score
df1[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].corr()


# In[23]:


#Correlation of EXT_SOURCE_1 and all other columns
df1.corr()['EXT_SOURCE_1'].sort_values(ascending=False)


# In[24]:


#Correlation of EXT_SOURCE_2 and all other columns
df1.corr()['EXT_SOURCE_2'].sort_values(ascending=False)


# In[25]:


#Correlation of EXT_SOURCE_3 and all other columns
df1.corr()['EXT_SOURCE_3'].sort_values(ascending=False)


# In[26]:


# Fill NA for Ext_source_1 with mean
df1['EXT_SOURCE_1']= df1['EXT_SOURCE_1'].fillna(df1['EXT_SOURCE_1'].mean(), axis=0).values


#sanity check 
df1['EXT_SOURCE_1'].isnull().sum()


# In[27]:


# Fill NA for EXT_source_2 with median
df1['EXT_SOURCE_2']= df1['EXT_SOURCE_2'].fillna(df1['EXT_SOURCE_2'].median(), axis=0).values


#Sanity check 
df1['EXT_SOURCE_2'].isnull().sum()


# In[28]:


# Fill NA for EXT_source_3 with median
df1['EXT_SOURCE_3']= df1['EXT_SOURCE_3'].fillna(df1['EXT_SOURCE_3'].median(), axis=0).values


#Sanity check 
df1['EXT_SOURCE_3'].isnull().sum()


# In[29]:


#Perform the exact same method for Ext_source 1, 2, 3 in test data

#Check the missing data for EXT_SOURCE 1,2,3 in app_test 1
df1[['EXT_SOURCE_1', 'EXT_SOURCE_2','EXT_SOURCE_3']].isnull().sum()


# For the next section, I will combine the train and test data together for easier cleaning data. After clean the data i will separate out the train vs test data. 

# In[30]:


#recheck the missing data summary
total_miss1 = df1.isnull().sum().sort_values(ascending=False)
miss_percent1 = (df1.isnull().sum()/df1.isnull().count()*100).sort_values(ascending=False)
miss_summ1 = pd.concat([total_miss1, miss_percent1], axis=1, keys=['Total Missing','Missing Percent'])
miss_summ1[miss_summ1['Missing Percent']>0]


# ### Occupation type
# 
# Another significant missing data is occupation type. Lets check the number of different occupation count and the percentage of each occupation. 

# In[31]:


print(df1['OCCUPATION_TYPE'].isnull().sum())
print(df1['OCCUPATION_TYPE'].shape)


# In[32]:


#Check the distribution of Occupation type
occupation_count = df1['OCCUPATION_TYPE'].value_counts()
occupation_percent = (df1['OCCUPATION_TYPE'].value_counts()/df1['OCCUPATION_TYPE'].count())*100
occupation_summary = pd.concat([occupation_count, occupation_percent], axis=1, keys =['Occupation Count', 'Occupation Percentage'])
occupation_summary


# Based on the occupation count, we can combine the type of occupation into following four categories: Laborers, Sales Staff, Core Staff, and Managers. Merged other occupation into the four categories based on the name of the occupation, since other occupations have less than 10% of the total data.
# 
# Laborers include: Drivers, Security Staff, Cooking staff, Cleaning Staff, Low-skill laborers,Waiters/barmen staff, Private service staff
# 
# Sales staff include:Realty Agents
# 
# Core staff include: High skill tech staff, accountants, Medicine staff, Secretaries, HR staff, IT staff
# 

# In[33]:


#Get the occupation summary index
occupation_summary.index


# In[34]:


#Recategorized laborers category based on the description
#Laborers include: Drivers, Security Staff, Cooking staff, Cleaning Staff, 
#Low-skill laborers,Waiters/barmen staff, Private service staff

df1.loc[df1['OCCUPATION_TYPE'].isin(['Drivers', 'Security staff','Cooking staff','Cleaning staff', 
                                                   'Private service staff', 'Low-skill Laborers', 
                                                   'Waiters/barmen staff']), 'OCCUPATION_TYPE']='Laborers'

#Recategorized Sales staff to include:Realty Agents
df1.loc[df1['OCCUPATION_TYPE'].isin(['Realty agents']),'OCCUPATION_TYPE']='Sales staff'

#Recategorized Core staff to include: High skill tech staff, accountants, Medicine staff, Secretaries, HR staff, IT staff
df1.loc[df1['OCCUPATION_TYPE'].isin(['High skill tech staff', 'Accountants', 'Medicine staff', 'Secretaries',
                                                   'HR staff', 'IT staff']),'OCCUPATION_TYPE']='Core staff'


# In[35]:


#Re-Check the distribution of Occupation type
occupation_count = df1['OCCUPATION_TYPE'].value_counts()
occupation_percent = (df1['OCCUPATION_TYPE'].value_counts()/df1['OCCUPATION_TYPE'].count())
occupation_summary = pd.concat([occupation_count, occupation_percent], axis=1, keys =['Occupation Count', 'Occupation Distribution'])
occupation_summary


# After recategorized the occupation type, there are still 111K of records still missing occupation. I assume the missing occupation type is missing at random, therefore fill in the missing values based on above distributions. 
# 

# In[36]:


#Fill in the na value by the same amount of percentage of current occupation distribution
df1['OCCUPATION_TYPE'].value_counts(normalize=True, dropna=False)


# In[37]:


#fill missing occupation with the four predefined categories and distribution percentage
df1['OCCUPATION_TYPE'] = df1['OCCUPATION_TYPE'].fillna(pd.Series(np.random.choice(['Laborers', 'Core staff', 'Sales staff',
                                                                                  'Managers'],p=[0.458530, 0.283515, 
                                                                                                 0.155830, 0.102125], size=len(df1))))


# In[38]:


#sanity check
occupation_count1 = df1['OCCUPATION_TYPE'].value_counts()
occupation_percent1 = (df1['OCCUPATION_TYPE'].value_counts()/df1['OCCUPATION_TYPE'].count())
occupation_summary1 = pd.concat([occupation_count1, occupation_percent1], axis=1, keys =['Occupation Count', 'Occupation Distribution'])
occupation_summary1


# In[39]:


print(df1['OCCUPATION_TYPE'].isnull().sum())
print(df1['OCCUPATION_TYPE'].shape)


# In[40]:


#recheck the missing data summary
total_miss1 = df1.isnull().sum().sort_values(ascending=False)
miss_percent1 = (df1.isnull().sum()/df1.isnull().count()*100).sort_values(ascending=False)
miss_summ1 = pd.concat([total_miss1, miss_percent1], axis=1, keys=['Total Missing','Missing Percent'])
miss_summ1[miss_summ1['Missing Percent']>0]


# ### AMT req credit bureau day, hour, month, quarter, week, year
# 
# The next missing items I will be dealing with are related to the columns AMT_REQ_CREDIT_BUREAU_DAY/HOUR/MON/QRT/WEEK/YEAR. According to the data description, the columns are measure number of inquiries to Credit Bureau about the client one hour/one month/one quarter/one week/one year before application. This seems important to our analysis. Based on common sense we know that the more inquiries you have on your credit file the less your credit score will be. Therefore, these numbers might be some important features to the prediction. Lets explore the distribution and correlation of the data first. 

# In[73]:


#Check the correlation for these columns
df1[['AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_MON',
     'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_YEAR']].corr()


# In[84]:


#check correlations with other columns
correlations = df1.corr()[['AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON',
           'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_YEAR']]

#visulized correlation matrix in heat map
plt.figure(figsize=(20,20))
sns.heatmap(correlations, vmin=-1, vmax=1,cmap="PiYG_r")
plt.show()


# In[66]:


#Explore AMT_req_credit_bureau_day
df1['AMT_REQ_CREDIT_BUREAU_DAY'].value_counts(normalize=True,dropna=False)


# In[67]:


#Explore AMT_req_credit_bureau_hour
df1['AMT_REQ_CREDIT_BUREAU_HOUR'].value_counts(normalize=True,dropna=False)
         


# In[68]:


#Explore AMT_req_credit_bureau_MON
df1['AMT_REQ_CREDIT_BUREAU_MON'].value_counts(normalize=True, dropna=False)


# In[69]:


#Explore AMT_req_credit_bureau_QRT
df1['AMT_REQ_CREDIT_BUREAU_QRT'].value_counts(normalize=True,dropna=False)


# In[70]:


#Explore AMT_req_credit_bureau_week
df1['AMT_REQ_CREDIT_BUREAU_WEEK'].value_counts(normalize=True,dropna=False)


# In[71]:


#Explore AMT_req_credit_bureau_year
df1['AMT_REQ_CREDIT_BUREAU_YEAR'].value_counts(normalize=True,dropna=False)


# Based on the results above, there are no strong correlations be different time period for the credit inquiries and there are no strong correlations with other features. In order to not distort the distribution of the filled missing values, it is better to fill the missing values by the same distribution of the current numbers. 

# In[95]:


#Fill NA by the distribution 

# get the ratio distribution of day, month, qrt, week, and year in the orignal data without na
r_day = df1['AMT_REQ_CREDIT_BUREAU_DAY'].value_counts(normalize=True)
r_hour = df1['AMT_REQ_CREDIT_BUREAU_HOUR'].value_counts(normalize=True)
r_mon = df1['AMT_REQ_CREDIT_BUREAU_MON'].value_counts(normalize=True)
r_qrt = df1['AMT_REQ_CREDIT_BUREAU_QRT'].value_counts(normalize=True)
r_week = df1['AMT_REQ_CREDIT_BUREAU_WEEK'].value_counts(normalize=True)
r_year = df1['AMT_REQ_CREDIT_BUREAU_YEAR'].value_counts(normalize=True)


# In[110]:


#Fill NA in AMT_REQ_CREDIT_BUREAU_DAY
df1['AMT_REQ_CREDIT_BUREAU_DAY'] = df1['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(pd.Series(
    np.random.choice(r_day.index, p=r_day.values, size=len(df1))))

#Fill NA in AMT_REQ_CREDIT_BUREAU_HOUR
df1['AMT_REQ_CREDIT_BUREAU_HOUR'] = df1['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(pd.Series(
    np.random.choice(r_hour.index, p=r_hour.values, size=len(df1))))

#Fill NA in AMT_REQ_CREDIT_BUREAU_MON
df1['AMT_REQ_CREDIT_BUREAU_MON'] = df1['AMT_REQ_CREDIT_BUREAU_MON'].fillna(pd.Series(
    np.random.choice(r_mon.index, p=r_mon.values, size=len(df1))))

#Fill NA in AMT_REQ_CREDIT_BUREAU_QRT
df1['AMT_REQ_CREDIT_BUREAU_QRT'] = df1['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(pd.Series(
    np.random.choice(r_qrt.index, p=r_qrt.values, size=len(df1))))

#Fill NA in AMT_REQ_CREDIT_BUREAU_WEEK
df1['AMT_REQ_CREDIT_BUREAU_WEEK'] = df1['AMT_REQ_CREDIT_BUREAU_WEEK'].fillna(pd.Series(
    np.random.choice(r_week.index, p=r_week.values, size=len(df1))))

#Fill NA in AMT_REQ_CREDIT_BUREAU_YEAR
df1['AMT_REQ_CREDIT_BUREAU_YEAR'] = df1['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(pd.Series(
    np.random.choice(r_year.index, p=r_year.values, size=len(df1))))


# In[111]:


#sanity check 
col = ['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT',
      'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_YEAR']

for i in col:
    print(df1[i].isnull().sum())


# In[113]:


#recheck the missing data summary
total_miss3 = df1.isnull().sum().sort_values(ascending=False)
miss_percent3 = (df1.isnull().sum()/df1.isnull().count()*100).sort_values(ascending=False)
miss_summ3 = pd.concat([total_miss3, miss_percent3], axis=1, keys=['Total Missing','Missing Percent'])
miss_summ3[miss_summ3['Missing Percent']>0]


# ### Name_Type_Suite
# 
# Name_type_suite is the measurement of who was accompany client when he was applying for the loan. The data dictionary did not specify whether it means taking a co-signer for the loan. Therefore, we will simply assume it is just who accompany the client to the branch at the time of applying the loan. Lets explore the data first. 

# In[144]:


#check the distribution of the Name_Type_Suite
df1['NAME_TYPE_SUITE'].value_counts(dropna=False)


# Based on the results, we can further recategorized this column by adding group of people to family and distribute NaN and other_B, other_A to other groups based on the current distribution ratio.

# In[170]:


#recategorized Group of people to Family
df1.loc[df1['NAME_TYPE_SUITE'].isin(['Group of people']), 'NAME_TYPE_SUITE']='Family'

#Change other_b and other_A to NAN
df1.loc[df1['NAME_TYPE_SUITE'].isin(['Other_B', 'Other_A']), 'NAME_TYPE_SUITE']=np.nan 

#Sanity check 
df1['NAME_TYPE_SUITE'].value_counts(dropna=False)


# In[173]:


#Redistribute NaN to other categories based on current ratio

r_name = df1['NAME_TYPE_SUITE'].value_counts(normalize=True, dropna=False)
df1['NAME_TYPE_SUITE'] = df1['NAME_TYPE_SUITE'].fillna(pd.Series(
    np.random.choice(r_name.index, p = r_name.values, size=len(df1))))

# Sanity Check 
df1['NAME_TYPE_SUITE'].value_counts(dropna=False)


# In[174]:


#recheck the missing data summary
total_miss3 = df1.isnull().sum().sort_values(ascending=False)
miss_percent3 = (df1.isnull().sum()/df1.isnull().count()*100).sort_values(ascending=False)
miss_summ3 = pd.concat([total_miss3, miss_percent3], axis=1, keys=['Total Missing','Missing Percent'])
miss_summ3[miss_summ3['Missing Percent']>0]


# ### Data missing less than 1%
# 
# Next we will deal with missing data that are missing less than 1% of the whole data. Since they are missing immaterial, I will fill in the missing values by either mean or median based on the distribution. 
# 
# 1. DEF/OBS_30/60_CNT_SOCIAL_CIRCLE. According to the data dictionary, OBS_30/60_CNT_SOCIAL_CIRCLE are the measurement of client's social surroundings with observable 30/60 DPD (days past due) default. DEF_30/60_CNT_SOCIAL_CIRCLE are the measurement of client's social surroundings defaulted on 30/60 DPD (days past due). 

# In[141]:


df1[['DEF_30_CNT_SOCIAL_CIRCLE','OBS_30_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE']].describe()


# In[175]:


# check the median
df1[['DEF_30_CNT_SOCIAL_CIRCLE','OBS_30_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE']].median()


# In[176]:


#fill in na with median
df1['DEF_30_CNT_SOCIAL_CIRCLE']= df1['DEF_30_CNT_SOCIAL_CIRCLE'].fillna(df1['DEF_30_CNT_SOCIAL_CIRCLE'].median(),axis=0).values
df1['OBS_30_CNT_SOCIAL_CIRCLE']= df1['OBS_30_CNT_SOCIAL_CIRCLE'].fillna(df1['OBS_30_CNT_SOCIAL_CIRCLE'].median(), axis=0).values
df1['DEF_60_CNT_SOCIAL_CIRCLE']= df1['DEF_60_CNT_SOCIAL_CIRCLE'].fillna(df1['DEF_60_CNT_SOCIAL_CIRCLE'].median(),axis=0).values
df1['OBS_60_CNT_SOCIAL_CIRCLE']= df1['OBS_60_CNT_SOCIAL_CIRCLE'].fillna(df1['OBS_60_CNT_SOCIAL_CIRCLE'].median(), axis=0).values

#recheck the missing data summary
total_miss3 = df1.isnull().sum().sort_values(ascending=False)
miss_percent3 = (df1.isnull().sum()/df1.isnull().count()*100).sort_values(ascending=False)
miss_summ3 = pd.concat([total_miss3, miss_percent3], axis=1, keys=['Total Missing','Missing Percent'])
miss_summ3[miss_summ3['Missing Percent']>0]


# 2. AMT_GOODS_PRICE is the measurement of the price of the goods for which the loan is given
# 
# 3. AMT_ANNUITY is the loan annuity.
# 
# 4. CNT_FAM_MEMBERS is how many family member does the client have
# 
# 5. DAYS_LAST_PHONE_CHANGE is how many days since last time client change phone number

# In[181]:


#Check the distribution

cols= ['AMT_GOODS_PRICE','AMT_ANNUITY','CNT_FAM_MEMBERS','DAYS_LAST_PHONE_CHANGE']

for i in cols:
    plt.figure()
    plt.hist(df1[i], edgecolor='black')
    plt.title(f'{i} Historgram')
    
plt.show()


# In[186]:


#Calculate the summary stats and median value
df1[cols].mean()


# In[187]:


df1[cols].median()


# Since all of the distribution are not symmetric, fill the NA with median for all of the four columns. 

# In[192]:


#fill na with median value in the four columns

for i in cols:
    df1[i] = df1[i].fillna(df1[i].median(), axis=0).values


# In[191]:


#recheck the missing data summary
total_miss4 = df1.isnull().sum().sort_values(ascending=False)
miss_percent4 = (df1.isnull().sum()/df1.isnull().count()*100).sort_values(ascending=False)
miss_summ4 = pd.concat([total_miss4, miss_percent4], axis=1, keys=['Total Missing','Missing Percent'])
miss_summ4[miss_summ4['Missing Percent']>0]


# The Target column has missing, which is the testing set that we need to predict later. So it is ok to leave it missing. All of the main data training and testing have been cleared of missing data. 

# In[195]:


df1.to_csv('Data/cleaned_df.csv', index=False)


# In[2]:


df1 = pd.read_csv('Data/cleaned_df.csv')


# ## Duplicate data

# In[7]:


#check duplicate rows 
df1.duplicated().sum()


# ## Explore the data

# ### Examine the columns with category values

# In[12]:


#Check df1 which columns are category values 
df1.head()


# In[13]:


df1.columns


# In[19]:


#loop over category columns to check the distribution 
cat_columns=['CODE_GENDER','FLAG_CONT_MOBILE', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 
             'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 
             'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 
             'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_EMAIL', 
             'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY','FLAG_PHONE', 'FLAG_WORK_PHONE', 
             'LIVE_CITY_NOT_WORK_CITY', 'LIVE_REGION_NOT_WORK_REGION', 'NAME_CONTRACT_TYPE', 'NAME_EDUCATION_TYPE', 
             'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE', 'NAME_TYPE_SUITE','OCCUPATION_TYPE', 
             'ORGANIZATION_TYPE', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'REG_CITY_NOT_LIVE_CITY', 
             'REG_CITY_NOT_WORK_CITY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION','TARGET', 
             'WEEKDAY_APPR_PROCESS_START']

for j in cat_columns:
    #plot the frequency
    df1[j].value_counts().plot(kind='bar')
    plt.title(f'{j} Bar Chart')
    plt.show()
    


# ### Convert the categorical columns to dummy 

# #### Code Gender
# 
# In the code gender columns, there are 4 rows marks with xna. Since it is very small number, we will just change these four to Male category. In addition, we will change Female to 0 and Male to 1 as categorical variables

# In[24]:


#Convert male to 1 and female to 0

df1['CODE_GENDER'].replace(to_replace={'F':0, 'M':1, 'XNA':1}, inplace=True)

#sanity check on gender
df1['CODE_GENDER'].value_counts()


# #### FLAG_OWN_CAR, FLAG_OWN_REALTY
# 
# Change Y to 1 and N to 0

# In[28]:


#flag_own_car y to 1 n to 0
df1['FLAG_OWN_CAR'].replace(to_replace={'Y':1,'N':0}, inplace=True)

#flag_own_realty y to 1 n to 0
df1['FLAG_OWN_REALTY'].replace(to_replace={'Y':1,'N':0}, inplace=True)

#sanity check 
display(df1['FLAG_OWN_CAR'].value_counts())
display(df1['FLAG_OWN_REALTY'].value_counts())


# #### Name Contract type 
# Cash loans to 1 and Revolving loans to 0

# In[32]:


#Name Contract type Cash loans to 1 and Revolving loans to 0
df1['NAME_CONTRACT_TYPE'].replace(to_replace={'Cash loans':1,'Revolving loans':0}, inplace=True)

#Sanity Check
df1['NAME_CONTRACT_TYPE'].value_counts()


# #### Explore the Organzation type
# 
# Change XNA to Other in Organization type.

# In[55]:


#replace XNA to Other 
df1['ORGANIZATION_TYPE'].replace(to_replace={'XNA':'Other'}, inplace=True)

#sanity check 
df1['ORGANIZATION_TYPE'].value_counts()


# #### One Hot Encoding
# 
# The following columns contains more than 2 categories, it is safer to use one hot encoding to replace the categorical value to dummy than label encoding, since they have no orders in the category:
# 
# NAME_EDUCATION_TYPE,NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, NAME_INCOME_TYPE, NAME_TYPE_SUITE, OCCUPATION_TYPE, WEEKDAY_APPR_PROCESS_START, ORGANIZATION TYPE

# In[57]:


#Change categorical columns to dummy with one-hot encoding
df2 = pd.get_dummies(df1)

display(df1.shape)
df2.shape


# ### Examine the columns with numeric values

# In[61]:


#loop over numeric columns to check the distribution 
num_columns=df1.columns[~df1.columns.isin(cat_columns)]

for i in num_columns:
    #plot the histgram
    plt.figure(figsize = (8,5))
    plt.hist(df2[i], edgecolor = 'black')
    plt.title(f'Distribution of {i}')
    plt.show()
  


# ### Correlation
# 
# Lets check whether there are some correlations related to the Target. Based on the result, there is no strong correlation between target and other features. 

# In[78]:


#Get the correlations of target vesus each individual columns
corr = df2.corr()

corr['TARGET'].sort_values(ascending=False)


# # Modeling
# 
# Before we start to modeling, we need to separate out the testing data we combined before since the testing data has no target column and we need to predict that particular columns. Training data already has target column and we will use that to develop our model and use the model to predict the target in the testing data. 
# 
# Since this problem is to predict 1 or 0, we need to use classification method (logistic regression, decision tree, random forest, SVM, xgboost, etc) to do the prediction. I will first run default machine learning method and then apply different hyperparameter to tunning different models. Evaluate each model and choose a best model to predict the separate test data. 

# ## Separate the testing data

# In[4]:


#Check whether the testing data before and after have the same rows
display(df2[df2['TARGET'].isnull()].shape)
display(app_test.shape)


# In[5]:


#Separate out testing data 
app_test2 = df2[df2['TARGET'].isnull()].reset_index()

#sanity check 
print(f'app_test2 Shape: {app_test2.shape}')
print(f'NaN values in app_test2 TARGET Column: {app_test2["TARGET"].value_counts(dropna=False).values} ')


# In[6]:


# get the trainging data from df
app_train2 = df2[df2['TARGET'].notnull()].reset_index()

#sanity check 
print(f'app_train2 Shape: {app_train2.shape}')
print(f'app_train Shape: {app_train.shape}')


# ## Train-Test Split
# 
# We need to split the training data into training and testing data for developing models. Also from the previous data exploration, we know that the target distribution in the training data is highly imbalanced. Therefore, we need to use stratisfied sampling method for any of the machine learning models.  

# In[7]:


#setting up SK_ID_CURR as index
app_train3 = app_train2.set_index('SK_ID_CURR')

#drop TARGET in app_train3 for setting up X 
X = app_train3.drop(['TARGET'], axis=1)
y = app_train2['TARGET']


# In[8]:


#Sanity check on X and y
print(f'Shape for app_train2: {app_train2.shape}')
print(f'Shape for app_train3: {app_train3.shape}')
print(f'Shape for X: {X.shape}')
print(f'Shape for y: {y.shape}')


# In[9]:


# check y's imbalance distribution
y.value_counts()


# In[10]:


#Startisfy sampling x and y 
from imblearn.over_sampling import SMOTE

X1, y1 = SMOTE().fit_resample(X,y)

#Check for y1 and X1 shape
print(f'Shape for X1: {X1.shape}')
print(f'Shape for y1: {y1.shape}')


# In[11]:


#check y1 whether balance after resampling
a =  pd.DataFrame(y1)
a.columns=['y1']
display(a['y1'].value_counts())


# In[12]:


#train-test split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train, y_test = train_test_split(X1,y1, test_size=0.2, random_state =1 )


# In[13]:


#import packages
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier


# ## Logistic Regression

# ### Simple Logistic regression
# 
# Lets start modeling with simple logistic regression with all of the default setting

# In[14]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression().fit(X_train,y_train)

#Score the result
print(f'log_reg score on training data: {log_reg.score(X_train,y_train)}')
print(f'log_reg score on testing data: {log_reg.score(X_test, y_test)}')


# ### Hyperparameter tunning for Logistic Regression
# 
# Using cross validation method to optimize hyperparameter c for logistic regression

# In[15]:


#using 5 fold cross validation to optimize hyperparameter for c
from sklearn.model_selection import cross_val_score

#creating empty list for cross validation score
cv_score = [] 

#create range for c to optimize
c_range =np.array([.00000001,.0000001,.000001,.00001,.0001,.001,.1,                1,10,100,1000,10000,100000,1000000,10000000,100000000,1000000000])

#Using for loops to create the iteration for 5 fold
for c in c_range:
    # Instanitate model
    logreg2 = LogisticRegression(C=c, n_jobs=-1)
    #Fit model on 5 folds.
    score = cross_val_score(logreg2, X_train, y_train, cv = 5).mean()
    cv_score.append(score)


# In[23]:


#visulize the cv score
plt.figure(figsize=(15,5))
plt.plot(c_range, cv_score, label='Cross Validation Score', marker='.')
plt.legend()
plt.xscale('log')
plt.xlabel('Regularization Parameter: C')
plt.xticks(c_range)
plt.ylabel('Cross Validation Score')
plt.title('Finding the Optimal C-value with Cross Validation ')
plt.grid()
plt.show()

which_max = np.array(cv_score).argmax()

print("The best model has C = ",c_range[which_max])


# Based on the cross validation, the logistic regression has the highest accuracy rate when parameter c= 100000. Therefore, c=100000 will be the best models for logistic regression.

# In[38]:


#print the score with the parameter c in tesing data 
log_reg3 = LogisticRegression(C=100000).fit(X_train,y_train)

#Score the result
print(f'Best Logistic Regression model score on training data: {"%.4f" % log_reg3.score(X_train,y_train)}')
print(f'Best Logistic Regression model score on testing data: {"%.4f" % log_reg3.score(X_test, y_test)}')


# ### Hyperparameter tunning for Logistic Regression after PCA reduction
# 
# This time I will use Standard Scaler and PCA dimension reduction before fitting to logistic regression and optimize the hyperparameter. Since I need to optimize hyperparameter for PCA and c for logistic regression, I will use gridsearch to optimize both at the same time. 
# 

# In[82]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#scaling the data
scaler =StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)


# In[83]:


#Perform PCA with defalut setting
my_pca = PCA().fit(X_train_s)
X_train_pca = my_pca.transform(X_train_s)


# In[90]:


#Visulize the number of component for cumulative line plot on the variance ratio

y_range =np.arange(0.1,1.05,0.05)
x_range =np.arange(0,165,5)
plt.figure(figsize=(20,8))
plt.plot(np.cumsum(my_pca.explained_variance_ratio_)) 
plt.xlabel('Principal component')
plt.ylabel('Explained Variance %')
plt.yticks(y_range)
plt.xticks(x_range)
plt.title('Number of Principal Component vs the Explained Variance %')
plt.grid()
plt.show()


# Based on the Number of Component vs the explained variance ratio percentage, we can see that between 135 and 140 components covered more than 95% of the data. Therefore, we will use number of component between 135 to 140 and c range value to find the best estimator. 

# In[92]:


# Create a pipeline of three steps. First, standardize the data. 

pipe = Pipeline([('scale', StandardScaler()), ('pca', PCA()), ('model', LogisticRegression())])

# Create Parameter-grid
param_grid=[{'scale':[StandardScaler(),None], 'pca':[PCA()], 'model':[LogisticRegression()],
            'pca__n_components':list(range(135,141,1)),
            'model__C':list(c_range)}
           ]


# In[94]:


# Conduct Parameter Optmization With Pipeline
# Create a grid search with 3-fold cross validataion
grid = GridSearchCV(pipe, param_grid, n_jobs=-1,cv=3,verbose=10)


# In[95]:


# Fit the grid search in train
fittedgrid = grid.fit(X_train_pca, y_train)


# In[98]:


# View The Best Parameters
print('Best C:', fittedgrid.best_estimator_.get_params()['model__C'])
print('Best Number Of Components:', fittedgrid.best_estimator_.get_params()['pca__n_components'])
print(); print(fittedgrid.best_estimator_.get_params()['model'])


# In[104]:


#Score in test data 
X_test_pca = my_pca.transform(X_test_s)
"%.3f" % fittedgrid.score(X_test_pca, y_test)


# In[105]:


#Perform PCA with 140 component and score the final model 

my_pca1 = PCA(n_components=140).fit(X_train_s)
X_train_pca1 = my_pca1.transform(X_train_s)
X_test_pca1 = my_pca1.transform(X_test_s)

#Score the final logistic regression with C=1
log_reg4 = LogisticRegression(C=1).fit(X_train_pca1, y_train)


print(f'log_reg4 score on training data: {"%.3f" % log_reg4.score(X_train_pca1,y_train)}')
print(f'log_reg4 score on testing data: {"%.3f" % log_reg4.score(X_test_pca1, y_test)}')


# ## Decision Tree

# ### Decision Tree with defalut setting 

# In[28]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier().fit(X_train,y_train)

#Score the result
print(f'dt score on training data: {"%.2f" % dt.score(X_train,y_train)}')
print(f'dt score on testing data: {"%.2f" % dt.score(X_test, y_test)}')


# ### Hyperparameter tunning for Decision Tree
# 
# Based on the simple decision tree result, we can see that there is clearly over fitting in the training set. Therefore, we will use cross validation to tunning the hyperparameter. 
# 
# In decision tree, we can adjust the max_depth, criterion, and splitter for the hyperparameter optimization. I will use the gridsearchcv with cross validation to optimize the decision tree
# 

# In[117]:


#get the max_depth from the defalut setting
dt.tree_.max_depth


# In[131]:


dt


# In[134]:


# Setting up depth of tree from 1 to 45 
tree_depth = list(range(1,46,1))

#setting up criterion option
criteria = ['gini','entropy']

#setting up splitter option 
splitter = ['best','random']

#create a pipeline of dt model

pipe_dt = Pipeline([('model_dt', DecisionTreeClassifier())])

#create parameter_grid
param_grid_dt = [{'model_dt': [DecisionTreeClassifier()],
                 'model_dt__max_depth':tree_depth,
                 'model_dt__criterion':criteria,
                 'model_dt__splitter':splitter}]

#Create grid search with 3-fold cross validation
grid_dt = GridSearchCV(pipe_dt, param_grid_dt, n_jobs=-1, cv=3, verbose=10)


# In[135]:


#fit the grid search in training set
fittedgrid_dt = grid_dt.fit(X_train,y_train)


# In[136]:


# View The Best Parameters
print('Best Max_depth:', fittedgrid_dt.best_estimator_.get_params()['model_dt__max_depth'])
print('Best Criteria:', fittedgrid_dt.best_estimator_.get_params()['model_dt__criterion'])
print('Best Splitter:', fittedgrid_dt.best_estimator_.get_params()['model_dt__splitter'])
print(); print(fittedgrid_dt.best_estimator_.get_params()['model_dt'])


# Based on above result, the best max_depth is 15 and the other two used default settings will have the ultimate results. Therefore, I will run the decision tree model with the hyperparameter get from the Gridsearchcv

# In[37]:


#Final Decision Tree model with max_depth=15

dt_2 = DecisionTreeClassifier(max_depth=15).fit(X_train, y_train)

#score the training and testing
print(f'DT_2 score on training: {"%0.4f" % dt_2.score(X_train, y_train)}')
print(f'DT_2 score on testing: {"%0.4f" % dt_2.score(X_test, y_test)}')


# ## Random Forest

# ### Random Forest with default setting 

# In[36]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier().fit(X_train, y_train)

#Score the result
print(f'rf score on training data: {"%.4f" % rf.score(X_train,y_train)}')
print(f'rf score on testing data: {"%.4f" % rf.score(X_test, y_test)}')


# In[19]:


#Get the default setting random forest 
rf


# In[16]:


#get the max_depth of the the random forest
rf.n_estimators


# ### Hyperparameter Tunning for Random Forest
# 
# In the Randomforest model we can control the following main parameters to improve the models:
# 
# 1. n_estimators
# 2. max_depth
# 3. min_samples_split 
# 
# To avoid long time of running in gridsearchcv, I will first use regular cross validation method to narrow down a range for each hyperparameter, then I will use gridsearchcv to find the optimized combination. 
# 

# In[16]:


# use cv to narrow down n_estimators

#create empty list for rf cv score
cv_score_rf = []

#create range for n_estimators
estimators = [10,50,100,150,200,250,300,350,400,450,500,550,600]

#Using for loops to create the iteration for 3 fold
for n in estimators:
    #Initiate randomforest
    rf2 = RandomForestClassifier(n_estimators=n, n_jobs=-1)
    #Fit model on 3 folds.
    score_rf = cross_val_score(rf2, X_train, y_train, cv = 3).mean()
    cv_score_rf.append(score_rf)


# In[18]:


#visulize the cv score
plt.figure(figsize=(15,5))
plt.plot(estimators, cv_score_rf, label='Cross Validation Score', marker='.')
plt.legend()
plt.xlabel('Regularization Parameter: n_estimator')
plt.xticks(estimators)
plt.ylabel('Cross Validation Score')
plt.title('Finding the Optimal n_estimator with Cross Validation for Randomforest ')
plt.grid()
plt.show()

which_max = np.array(cv_score_rf).argmax()

print("The best model has number of trees ",estimators[which_max])


# In[20]:


#Using Gridsearchcv to find other best parameters with 450 trees

# Setting up depth of tree each from 1 to 60
depth = [1,5,10,15,20,25,30,40,45,50,55,60,None]

#setting up slit range 
split = [2,5,10,15,20,25,30]

#create a pipeline of rf model

pipe_rf = Pipeline([('model_rf', RandomForestClassifier())])

#Create parameter_grid

param_grid_rf = [{'model_rf': [RandomForestClassifier()],
                 'model_rf__max_depth':depth,
                 'model_rf__n_estimators':[450],
                 'model_rf__min_samples_split':split}]

#Create grid search with 3-fold cross validation
grid_rf = GridSearchCV(pipe_rf, param_grid_rf, n_jobs=-1, cv=3, verbose=10)


# In[21]:


#fit the grid search in training set
fittedgrid_rf = grid_rf.fit(X_train,y_train)


# In[22]:


# View The Best Parameters
print('Best Max_depth:', fittedgrid_rf.best_estimator_.get_params()['model_rf__max_depth'])
print('Best Number of Estimators:', fittedgrid_rf.best_estimator_.get_params()['model_rf__n_estimators'])
print('Best Min Samples split:', fittedgrid_rf.best_estimator_.get_params()['model_rf__min_samples_split'])
print(); print(fittedgrid_rf.best_estimator_.get_params()['model_rf'])


# In[16]:


#Score the testing set with the hyperparameter

rf2 = RandomForestClassifier(n_estimators=450, n_jobs=-1, max_depth=60, min_samples_split=2).fit(X_train, y_train)

print(f'rf2 score on training data: {"%.4f" %rf2.score(X_train, y_train)}')
print(f'rf2 score on testing data: {"%.4f" % rf2.score(X_test, y_test)}')


# Based on the results of 450 estimators Random forest, it seems that that there is a little overfitting problem. I will change the number of estimators to 10, 50 and 200 to see if there are any difference. And will pick a best among the three. From the accuracy chart of optimize number of estimators, there is no significant accuracy difference when number of estimators equals to 10, 50, 200, and 450. 

# In[17]:


#Score the testing set with the hyperparameter when n_estimators =50

rf3 = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=60, min_samples_split=2).fit(X_train, y_train)

print(f'rf3 score on training data: {"%.4f" %rf3.score(X_train, y_train)}')
print(f'rf3 score on testing data: {"%.4f" % rf3.score(X_test, y_test)}')


# In[19]:


#Score the testing set with the hyperparameter when n_estimators = 200

rf4 = RandomForestClassifier(n_estimators=200, n_jobs=-1, max_depth=60, min_samples_split=2).fit(X_train, y_train)

print(f'rf4 score on training data: {"%.4f" %rf4.score(X_train, y_train)}')
print(f'rf4 score on testing data: {"%.4f" % rf4.score(X_test, y_test)}')


# In[20]:


#Score the testing set with the hyperparameter when n_estimators = 200

rf5 = RandomForestClassifier(n_estimators=10, n_jobs=-1, max_depth=60, min_samples_split=2).fit(X_train, y_train)

print(f'rf4 score on training data: {"%.4f" %rf5.score(X_train, y_train)}')
print(f'rf4 score on testing data: {"%.4f" % rf5.score(X_test, y_test)}')


# From the above results, we can see that there is no significant accuracy difference on the testing data, but when number of estimators equals to 10, the accuracy on training set is less overfitting compares to other number of estimators. Therefore, I will choose 10 as my number of estimators in random forest. 

# ## XGBOOST

# ### XGBoost with default setting

# In[21]:


from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_jobs = -1).fit(X_train, y_train)

print(f'Train Data Score for XGboost with X-y: {xgb_model.score(X_train, y_train)}')
print(f'Test Data Score for XGboost with X-y: {xgb_model.score(X_test, y_test)}')


# ### Hyperparameter optimize for XGBoost
# 

# In[16]:


# use cv to narrow down max_depth

#create empty list for xg cv score
cv_score_xg = []

#create range for n_estimators
depth = list(range(1,21,1))

#Using for loops to create the iteration for 3 fold
for d in depth:
    #Initiate XGboost
    xg2 = XGBClassifier(max_depth=d, n_jobs=-1)
    #Fit model on 3 folds.
    score_xg = cross_val_score(xg2, X_train, y_train, cv = 3).mean()
    cv_score_xg.append(score_xg)


# In[28]:


#visulize the cv score
plt.figure(figsize=(15,5))
plt.plot(depth, cv_score_xg, label='Cross Validation Score', marker='.')
plt.legend()
plt.xlabel('Regularization Parameter: max_depth')
plt.xticks(depth)
plt.ylabel('Cross Validation Score')
plt.title('Finding the Optimal depth with Cross Validation for XGboost')
plt.grid()
plt.show()

which_max = np.array(cv_score_xg).argmax()

print("The best model has max depth ",depth[which_max])


# In[19]:


# use cv to narrow down learning rate 

#create empty list for xg cv score
cv_score1_xg = []

#create range for n_estimators
learning = [0.0001,0.001,0.01,0.1,1,10,100,1000]

#Using for loops to create the iteration for 3 fold
for l in learning:
    #Initiate XGboost
    xg3 = XGBClassifier(learning_rate=l, n_jobs=-1)
    #Fit model on 3 folds.
    score1_xg = cross_val_score(xg3, X_train, y_train, cv = 3).mean()
    cv_score1_xg.append(score1_xg)


# In[29]:


#visulize the cv score
plt.figure(figsize=(15,5))
plt.plot(learning, cv_score1_xg, label='Cross Validation Score', marker='.')
plt.legend()
plt.xlabel('Regularization Parameter: learning rate')
plt.xticks(learning)
plt.xscale('log')
plt.ylabel('Cross Validation Score')
plt.title('Finding the Optimal learning rate with Cross Validation for XGboost')
plt.grid()
plt.show()

which_max = np.array(cv_score1_xg).argmax()

print("The best model has learning rate of ",learning[which_max])


# In[24]:


# use cv to narrow down number of estiator

#create empty list for xg cv score
cv_score2_xg = []

#create range for n_estimators
estimators = [1,10,50,100,200,300,400,500,1000]

#Using for loops to create the iteration for 3 fold
for e in estimators:
    #Initiate XGboost
    xg4 = XGBClassifier(n_estimators=e, n_jobs=-1)
    #Fit model on 3 folds.
    score2_xg = cross_val_score(xg4, X_train, y_train, cv = 3).mean()
    cv_score2_xg.append(score2_xg)


# In[27]:


#visulize the cv score
plt.figure(figsize=(15,5))
plt.plot(estimators, cv_score2_xg, label='Cross Validation Score', marker='.')
plt.legend()
plt.xlabel('Regularization Parameter: number of estimator')
plt.xticks(estimators)
plt.ylabel('Cross Validation Score')
plt.title('Finding the Optimal estimators with Cross Validation for XGboost')
plt.grid()
plt.show()

which_max = np.array(cv_score2_xg).argmax()

print("The best model has number of trees ",estimators[which_max])


# In[30]:


#Using Gridsearchcv to find the best parameters

#creating range for max_depth
xg_depth = [2,3,4,5]


# creating range for learning_rate
xg_learning =[0.1,1]

#creating range for n_estimators 
xg_estimators = [50,100]


#create a pipeline of SVM model

pipe_xg = Pipeline([('model_xg', XGBClassifier())])

#Create parameter_grid

param_grid_xg = [{'model_xg': [XGBClassifier()],
                 'model_xg__max_depth':xg_depth,
                 'model_xg__n_estimators':xg_estimators,
                 'model_xg__learning_rate':xg_learning}]

#Create grid search with 3-fold cross validation
grid_xg = GridSearchCV(pipe_xg, param_grid_xg, n_jobs=-1, cv=3, verbose=10)


# In[31]:


fittedgrid_xg = grid_xg.fit(X_train, y_train)


# In[32]:


# View The Best Parameters
print('Best Max_depth:', fittedgrid_xg.best_estimator_.get_params()['model_xg__max_depth'])
print('Best Number of Estimators:', fittedgrid_xg.best_estimator_.get_params()['model_xg__n_estimators'])
print('Best Learning rate:', fittedgrid_xg.best_estimator_.get_params()['model_xg__learning_rate'])
print(); print(fittedgrid_xg.best_estimator_.get_params()['model_xg'])


# In[34]:


#Score the train and test data based on the best parameter search from gridsearch cv
xg5 = XGBClassifier(n_jobs = -1, n_estimators=100, max_depth=5, learning_rate=0.1).fit(X_train, y_train)

print(f'final xgboost score on training data: {"%.4f" %xg5.score(X_train, y_train)}')
print(f'final xgboost score on testing data: {"%.4f" % xg5.score(X_test, y_test)}')


# The result is slightly better than the default settings and the result do not seemed to be overfitting. Below is a summary of all the models. 

# ## Final Result Comparison and Model Evaluation

# In[65]:


#Create DF for all models results
summary_result = pd.DataFrame({'Models': ['Logistic','Optimized Logistic','Decision Tree','Optimized Decision Tree',
                                          'Random Forest', 'Optimized Random Forest', 'XGboost','Optimized XGboost'],
                              'Accuracy':[0.5793,0.697,0.91,0.9357,0.9557,0.9553,0.9549,0.9562]})

summary_result = summary_result.sort_values('Accuracy')
summary_result


# Based on the results, the Random Forest and Optimized XGboost have the highest accuracy rating. Therefore, we will evaluate these two models with confusion matrix and ROC curve.  

# ### Confusion Matrix for Random Forest and Optimized XGboost

# #### Random Forest Confusion Matrix

# In[72]:


#import classification_report package
from sklearn.metrics import classification_report


# In[73]:


#get y_predict on test data
y_predict = rf.predict(X_test)

# Confusion Matrix for Random Forest
print(classification_report(y_test, y_predict))


# #### Optimized XGBoost Confusion Matrix

# In[74]:


#get y_predict on test data
y_predict_xg = xg5.predict(X_test)

# Confusion Matrix for Optimized XGBoost
print(classification_report(y_test, y_predict_xg))


# ### ROC Curve for Random Forest and Optimized XGboost

# In[82]:


from sklearn.metrics import roc_curve, roc_auc_score

#Get the predicted postive case only for Random forest
y_proba = rf.predict_proba(X_test)[:,1] 
fprs, tprs, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

#Get the predicted postive case only for Optimized XGboost
y_proba_xg = xg5.predict_proba(X_test)[:,1] 
fprs_xg, tprs_xg, thresholds_xg = roc_curve(y_test, y_proba_xg)
roc_auc_xg = roc_auc_score(y_test, y_proba_xg)

#plot the roc curve
plt.figure(figsize=(10,8))
plt.plot(fprs, tprs, color ='blue', lw=2, label='Random Forest AUC= %0.2f' % roc_auc)
plt.plot(fprs_xg, tprs_xg, color ='orange', lw=2, label='XGBoost AUC= %0.2f' % roc_auc_xg)
plt.plot([0,1],[0,1], color ='black', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC for Random Forest and XGboost Model')
plt.legend(loc ='best')
plt.show()

print(f'Area under curve (AUC):{roc_auc}')
print(f'Area under Curve (AUC- XG): {roc_auc_xg}')
print()


# Based on the above model evaluation, both Random Forest and Optimized XGboost have similar precision and recall score as well as AUC. Both models performs very well in terms of identify the target to be 1 or 0. The XGboost AUC is slightly higher than Random Forest. Therefore, I will choose XGboost model to predict the final test data, the dataset without the target identification. 

# In[99]:


#Sanity Check on app_test2
app_test2.shape


# In[100]:


#Sanity check on X_test
X_test.shape


# In[104]:


#view app_test2 head
app_test2.head()


# Based on the comparison, we need to drop the TARGET and set SK_ID_CURR as the new index. 

# In[166]:


#drop the current index
app_test3 = app_test2.drop(['TARGET'], axis=1)

#set SK_ID_CURR as new index
app_test4= app_test3.set_index('SK_ID_CURR')


# In[167]:


#Sanity check 
display(app_test2.shape)
display(app_test4.shape)
app_test4.head()


# Now we can use the optimized XGboost model to predict the target with the updated app_test4

# In[181]:


#predict the target using xg5 model on the test data without target
target_predtict = xg5.predict_proba(app_test4.values)


# In[183]:


target_predtict


# In[184]:


#sanity check 
target_predtict[:,1]


# In[186]:


#combined target_predict and app_test4 index into new dataframe
tx_predict = pd.DataFrame({'SK_ID_CURR': app_test4.index,
                          'TARGET':target_predtict[:,1]})


# In[187]:


tx_predict.head()


# In[188]:


#export the prediction to csv and sumbit to Kaggle for evaluation
tx_predict.to_csv('Data/tx_predict3.csv', index=False)


# Kaggle's score is 0.7253. The score could be improved with more extensive feature engineering, using other dataset provided to come up with more features. 
