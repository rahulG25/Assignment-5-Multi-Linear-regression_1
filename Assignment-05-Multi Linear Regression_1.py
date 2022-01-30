#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot


# In[2]:


# import dataset
data=pd.read_csv("50_Startups.csv")
data


# # EDA

# In[3]:


data.info()


# In[4]:


data1=data.rename({'R&D Spend':'RDS','Administration':'ADMS','Marketing Spend':'MKTS'},axis=1)
data1


# In[5]:


data1[data1.duplicated()] # No duplicated data


# In[6]:


data1.describe()


# # Correlation Analysis

# In[7]:


data1.corr()


# In[8]:


sns.set_style(style='darkgrid')
sns.pairplot(data1)


# # Model Building

# In[10]:


model=smf.ols("Profit~RDS+ADMS+MKTS",data=data1).fit()


# # Model Testing

# In[11]:


# Finding Coefficient parameters
model.params


# In[12]:


# Finding tvalues and pvalues
model.tvalues , np.round(model.pvalues,5)


# In[13]:


# Finding rsquared values
model.rsquared , model.rsquared_adj  # Model accuracy is 94.75%


# In[14]:


# Build SLR and MLR models for insignificant variables 'ADMS' and 'MKTS'
# Also find their tvalues and pvalues


# In[15]:


slr_a=smf.ols("Profit~ADMS",data=data1).fit()
slr_a.tvalues , slr_a.pvalues  # ADMS has in-significant pvalue


# In[16]:


slr_m=smf.ols("Profit~MKTS",data=data1).fit()
slr_m.tvalues , slr_m.pvalues  # MKTS has significant pvalue


# In[17]:


mlr_am=smf.ols("Profit~ADMS+MKTS",data=data1).fit()
mlr_am.tvalues , mlr_am.pvalues  # varaibles have significant pvalues


# # Model Validation

# # Two Techniques: 1. Collinearity Check & 2. Residual Analysis

# In[18]:


# 1) Collinearity Problem Check
# Calculate VIF = 1/(1-Rsquare) for all independent variables

rsq_r=smf.ols("RDS~ADMS+MKTS",data=data1).fit().rsquared
vif_r=1/(1-rsq_r)

rsq_a=smf.ols("ADMS~RDS+MKTS",data=data1).fit().rsquared
vif_a=1/(1-rsq_a)

rsq_m=smf.ols("MKTS~RDS+ADMS",data=data1).fit().rsquared
vif_m=1/(1-rsq_m)

# Putting the values in Dataframe format
d1={'Variables':['RDS','ADMS','MKTS'],'Vif':[vif_r,vif_a,vif_m]}
Vif_df=pd.DataFrame(d1)
Vif_df


# In[19]:


# None variable has VIF>20, No Collinearity, so consider all varaibles in Regression equation


# In[20]:


# 2) Residual Analysis
# Test for Normality of Residuals (Q-Q Plot) using residual model (model.resid)

sm.qqplot(model.resid,line='q')
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[21]:


list(np.where(model.resid<-30000))


# In[22]:


# Test for Homoscedasticity or Heteroscedasticity (plotting model's standardized fitted values vs standardized residual values)

def standard_values(vals) : return (vals-vals.mean())/vals.std()  # User defined z = (x - mu)/sigma


# In[23]:


plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show() 


# In[24]:


# Test for errors or Residuals Vs Regressors or independent 'x' variables or predictors 
# using Residual Regression Plots code graphics.plot_regress_exog(model,'x',fig)    # exog = x-variable & endog = y-variable


# In[25]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'RDS',fig=fig)
plt.show()


# In[26]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'ADMS',fig=fig)
plt.show()


# In[27]:


fig=plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'MKTS',fig=fig)
plt.show()


# # Model Deletion Diagnostics (checking Outliers or Influencers)

# # Two Techniques : 1. Cook's Distance & 2. Leverage value

# In[28]:


# 1. Cook's Distance: If Cook's distance > 1, then it's an outlier
# Get influencers using cook's distance
(c,_)=model.get_influence().cooks_distance
c


# In[29]:


# Plot the influencers using the stem plot
fig=plt.figure(figsize=(20,7))
plt.stem(np.arange(len(data1)),np.round(c,5))
plt.xlabel('Row Index')
plt.ylabel('Cooks Distance')
plt.show()


# In[30]:


# Index and value of influencer where C>0.5
np.argmax(c) , np.max(c)


# In[31]:


# 2. Leverage Value using High Influence Points : Points beyond Leverage_cutoff value are influencers
influence_plot(model)
plt.show()


# In[32]:


# Leverage Cuttoff Value = 3*(k+1)/n ; k = no.of features/columns & n = no. of datapoints
k=data1.shape[1]
n=data1.shape[0]
leverage_cutoff = (3*(k+1))/n
leverage_cutoff


# In[33]:


data1[data1.index.isin([49])] 


# # Improving the Model

# In[34]:


# Discard the data points which are influencers and reassign the row number (reset_index(drop=True))
data2=data1.drop(data1.index[[49]],axis=0).reset_index(drop=True)
data2


# # Model Deletion Diagnostics and Final Model

# In[37]:


while np.max(c)>0.5 :
    model=smf.ols("Profit~RDS+ADMS+MKTS",data=data2).fit()
    (c,_)=model.get_influence().cooks_distance
    c
    np.argmax(c) , np.max(c)
    data2=data2.drop(data2.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
    data2
else:
    final_model=smf.ols("Profit~RDS+ADMS+MKTS",data=data2).fit()
    final_model.rsquared , final_model.aic
    print("Thus model accuracy is improved to",final_model.rsquared)


# In[38]:


final_model.rsquared


# In[39]:


data2


# # Model Predictions

# In[40]:


# say New data for prediction is
new_data=pd.DataFrame({'RDS':70000,"ADMS":90000,"MKTS":140000},index=[0])
new_data


# In[41]:


# Manual Prediction of Price
final_model.predict(new_data)


# In[42]:


# Automatic Prediction of Price with 90.02% accurcy
pred_y=final_model.predict(data2)
pred_y


# # table containing R^2 value for each prepared model

# In[43]:


d2={'Prep_Models':['Model','Final_Model'],'Rsquared':[model.rsquared,final_model.rsquared]}
table=pd.DataFrame(d2)
table

