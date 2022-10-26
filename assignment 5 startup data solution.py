#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot.


# In[2]:


data=pd.read_csv(r"C:\Users\Lovely_Ray\Desktop\data science\Assignment 5\50_Startups.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data1=data.rename({'R&D Spend':'RDS','Administration':'ADS','Marketing Spend':'MKTS'},axis=1)
data1


# In[6]:


data1[data1.duplicated()]


# In[7]:


data1.corr() #correlation analysis


# In[9]:


sns.pairplot(data1)


# In[12]:


model=smf.ols('Profit~RDS+ADS+MKTS',data=data1).fit() #model building


# In[13]:


model.tvalues , np.round(model.pvalues,5) #model testing


# In[14]:


(model.rsquared,model.rsquared_adj) #rsquared value


# In[15]:


ml_a=smf.ols('Profit~ADS',data = data1).fit()  #SLR model
#t and p-Values
print(ml_a.tvalues, '\n', ml_a.pvalues) 


# In[16]:


ml_m=smf.ols('Profit~MKTS',data = data1).fit()  #SLR model
#t and p-Values
print(ml_m.tvalues, '\n', ml_m.pvalues) 


# In[17]:


ml_am=smf.ols('Profit~ADS+MKTS',data = data1).fit()  #SLR model
#t and p-Values
print(ml_am.tvalues, '\n', ml_am.pvalues) 


# In[18]:


rsq_a = smf.ols('ADS~RDS+MKTS',data=data1).fit().rsquared  
vif_a = 1/(1-rsq_a)

rsq_r = smf.ols('RDS~ADS+MKTS',data=data1).fit().rsquared  
vif_r = 1/(1-rsq_r)

rsq_m = smf.ols('MKTS~ADS+RDS',data=data1).fit().rsquared  
vif_m = 1/(1-rsq_m)

# Storing vif values in a data frame
d1 = {'Variables':['ADS','MKTS','RDS'],'VIF':[vif_a,vif_m,vif_r]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame #calculation of VIF


# In[ ]:


# None variable has VIF>20, so no Collinearity


# In[19]:


qqplot=sm.qqplot(model.resid,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")
plt.show() # Residual analysis


# In[20]:


list(np.where(model.resid<-30000))


# In[21]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std() #Residual spot for homoscedasticity


# In[22]:


plt.scatter(get_standardized_values(model.fittedvalues),
            get_standardized_values(model.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[23]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "ADS", fig=fig)
plt.show() #Residuals vs regressors


# In[24]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "MKTS", fig=fig)
plt.show()


# In[25]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "RDS", fig=fig)
plt.show()


# In[27]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance
c #model deletion diagnostics


# In[28]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(data1)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[29]:


(np.argmax(c),np.max(c)) #data influencer detection 


# In[30]:


influence_plot(model)
plt.show()


# In[32]:


k = data1.shape[1]
n = data1.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff


# In[33]:


data1[data1.index.isin([49])] # treatment of influencers


# In[34]:


data2=data1.drop(data1.index[[49]],axis=0).reset_index()


# In[35]:


data2=data2.drop(['index'],axis=1)
data2


# In[36]:


final_ml_A= smf.ols('Profit~ADS+RDS',data = data2).fit() #building model


# In[37]:


(final_ml_A.rsquared,final_ml_A.aic)


# In[38]:


final_ml_M= smf.ols('Profit~MKTS+RDS',data = data2).fit()


# In[39]:


(final_ml_M.rsquared,final_ml_M.aic) #model accuracy is 96%


# In[40]:


model_influence_V = final_ml_M.get_influence()
(c_V, _) = model_influence_V.cooks_distance


# In[41]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(data2)),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[42]:


(np.argmax(c_V),np.max(c_V)) # cook's distance is >0.5, so going with previous model accuracy value


# In[43]:


final_ml_M= smf.ols('Profit~MKTS+RDS',data = data2).fit()


# In[44]:


(final_ml_M.rsquared,final_ml_M.aic) #model accuracy 96%


# In[45]:


new_data=pd.DataFrame({'RDS':75000,"ADMS":96000,"MKTS":200000},index=[1]) #prediction for new data
new_data


# In[46]:


final_ml_M.predict(new_data)


# In[47]:


final_ml_M.predict(data2.iloc[0:5,])


# In[48]:


pred_y = final_ml_M.predict(data2)
pred_y # applying final model to the data.


# In[ ]:





# In[ ]:




