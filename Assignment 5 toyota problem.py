#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot


# In[5]:


toyota=pd.read_csv(r"C:\Users\Lovely_Ray\Desktop\data science\Assignment 5\ToyotaCorolla.csv",encoding='latin1')


# In[6]:


toyota.head()


# In[7]:


toyota.info()


# In[9]:


toyota1=pd.concat([toyota.iloc[:,2:4],toyota.iloc[:,6:7],toyota.iloc[:,8:9],toyota.iloc[:,12:14],toyota.iloc[:,15:18]],axis=1)
toyota1


# In[10]:


toyota1.info()


# In[11]:


toyota1.corr() #correlation analysis


# In[21]:


toyota2=toyota1.rename({'Age_08_04':'Age','cc':'CC','Quarterly_Tax':'QT'},axis=1)
toyota2


# In[22]:


toyota2[toyota2.duplicated()] # dropping duplicate rows


# In[23]:


toyota3=toyota2.drop_duplicates()
toyota3


# In[24]:


toyota3.corr() #correlation analysis


# In[25]:


sns.pairplot(toyota3)


# In[26]:


model=smf.ols('Price~Age+KM+HP+CC+Doors+Gears+QT+Weight',data=toyota3).fit() #model building


# In[30]:


model.tvalues, np.round(model.pvalues,5) #computation of tvalues & pvalues


# In[28]:


model.rsquared, model.rsquared_adj #finding rsquared values   #model accuracy is 86%.


# In[31]:


ml_cc=smf.ols("Price~CC",data = toyota3).fit() #building sub models for variable cc & doors. also computation of tvalues & pvalues
ml_cc.tvalues, ml_cc.pvalues


# In[32]:


ml_d=smf.ols("Price~Doors",data = toyota3).fit()
ml_d.tvalues, ml_d.pvalues


# In[33]:


ml_ccd=smf.ols("Price~Doors+CC",data = toyota3).fit()
ml_ccd.tvalues, ml_ccd.pvalues


# In[34]:


rsq_Age = smf.ols('Age~KM+HP+CC+Doors+Gears+QT+Weight',data=toyota3).fit().rsquared  
vif_Age = 1/(1-rsq_Age)
rsq_KM = smf.ols('KM~Age+HP+CC+Doors+Gears+QT+Weight',data=toyota3).fit().rsquared  
vif_KM = 1/(1-rsq_KM)
rsq_HP = smf.ols('HP~KM+Age+CC+Doors+Gears+QT+Weight',data=toyota3).fit().rsquared  
vif_HP = 1/(1-rsq_HP)
rsq_CC = smf.ols('CC~KM+HP+Age+Doors+Gears+QT+Weight',data=toyota3).fit().rsquared  
vif_CC = 1/(1-rsq_CC)
rsq_Doors = smf.ols('Doors~KM+HP+CC+Age+Gears+QT+Weight',data=toyota3).fit().rsquared  
vif_Doors = 1/(1-rsq_Doors)
rsq_Gears = smf.ols('Gears~KM+HP+CC+Doors+Age+QT+Weight',data=toyota3).fit().rsquared  
vif_Gears = 1/(1-rsq_Gears)
rsq_QT = smf.ols('QT~KM+HP+CC+Doors+Gears+Age+Weight',data=toyota3).fit().rsquared  
vif_QT = 1/(1-rsq_QT)
rsq_Weight = smf.ols('Weight~KM+HP+CC+Doors+Gears+QT+Age',data=toyota3).fit().rsquared  
vif_Weight = 1/(1-rsq_Weight)
d1 = {'Variables':['Age','KM','HP','CC','Doors','Gears','QT','Weight'],'VIF':[vif_Age,vif_KM,vif_HP,vif_CC,vif_Doors,vif_Gears,vif_QT,vif_Weight]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame  #calculation of VIF for colleniarity check. # none of the variables has VIF>20.so no colleniarity


# In[35]:


qqplot=sm.qqplot(model.resid,line='q') # line = 45 to draw the diagnoal line #Residual analysis, QQ Plot
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[36]:


list(np.where(model.resid>6000))


# In[37]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std() # Residual plot for Homoscadasticity


# In[38]:


plt.scatter(get_standardized_values(model.fittedvalues),
            get_standardized_values(model.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[39]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Age", fig=fig)
plt.show()  # test for errors or residuals vs regressors for each input variable.


# In[40]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "KM", fig=fig)
plt.show()


# In[41]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "HP", fig=fig)
plt.show()


# In[42]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "CC", fig=fig)
plt.show()


# In[43]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Doors", fig=fig)
plt.show()


# In[44]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Gears", fig=fig)
plt.show()


# In[45]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "QT", fig=fig)
plt.show()


# In[46]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "Weight", fig=fig)
plt.show()


# In[91]:


model_influence = model.get_influence()
(c, _) = model_influence.cooks_distance
c                                        #computation of cook's distance and finding leverage value


# In[48]:


fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(toyota3)), np.round(c, 3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[49]:


(np.argmax(c),np.max(c))


# In[50]:


influence_plot(model)
plt.show()


# In[51]:


k = toyota3.shape[1]
n = toyota3.shape[0]
leverage_cutoff = 3*((k + 1)/n)


# In[52]:


leverage_cutoff


# In[53]:


toyota3[toyota3.index.isin([80])] #improving the model by deleting the outlier


# In[92]:


toyota4=toyota3.copy()
toyota4


# In[102]:


toyotanew=toyota4.drop(toyota4.index[[80]],axis=0)


# In[103]:


toyotanew


# In[104]:


toyotafinal=toyotanew.reset_index()
toyotafinal


# In[105]:


toyotafinal1=toyotafinal.drop(['index'],axis=1)
toyotafinal1


# In[108]:


final_ml_cc= smf.ols('Price~Age+KM+HP+CC+Gears+QT+Weight',data = toyotafinal1).fit()


# In[109]:


(final_ml_cc.rsquared,final_ml_cc.aic) #model accuracy has been improved to 87%


# In[110]:


final_ml_d= smf.ols('Price~Age+KM+HP+Doors+Gears+QT+Weight',data = toyotafinal1).fit()


# In[111]:


(final_ml_d.rsquared,final_ml_d.aic)


# In[112]:


model_influence_V = final_ml_cc.get_influence()
(c_V, _) = model_influence_V.cooks_distance


# In[113]:


c


# In[114]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(toyotafinal1)),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[115]:


(np.argmax(c_V),np.max(c_V))


# In[116]:


influence_plot(model)
plt.show()


# In[132]:


toyotafinal2=toyotafinal1.drop(toyotafinal1.index[[219]],axis=0) #model improvement exercise


# In[133]:


toyotafinal2


# In[134]:


toyo1=toyotafinal2.reset_index()
toyo1


# In[135]:


toyo2=toyo1.drop(['index'],axis=1)
toyo2


# In[136]:


final_ml_cc= smf.ols('Price~Age+KM+HP+Doors+Gears+QT+Weight',data = toyo2).fit()


# In[137]:


model_influence_V = final_ml_cc.get_influence()
(c_V, _) = model_influence_V.cooks_distance


# In[138]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(toyo2)),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance');


# In[139]:


(np.argmax(c_V),np.max(c_V))


# In[140]:


influence_plot(model)
plt.show()


# In[141]:


toyo3=toyo2.drop(toyo2.index[[957]],axis=0)


# In[142]:


toyo3


# In[144]:


toyo4=toyo3.reset_index()


# In[145]:


toyo5=toyo4.drop(['index'],axis=1)
toyo5


# In[146]:


final_ml_cc= smf.ols('Price~Age+KM+HP+Doors+Gears+QT+Weight',data = toyo5).fit()


# In[147]:


(final_ml_cc.rsquared,final_ml_cc.aic) # final model accuracy coming at 87.12%


# In[148]:


model_influence_V = final_ml_cc.get_influence()
(c_V, _) = model_influence_V.cooks_distance


# In[149]:


fig= plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(toyo5)),np.round(c_V,3));
plt.xlabel('Row index')
plt.ylabel('Cooks Distance'); # as cook's distance now >0.5, so we are finiling model accuracy @87.12%


# In[150]:


new_data=pd.DataFrame({'Age':25,"KM":30000,"HP":90,"CC":1600,"Doors":3,"Gears":5,"QT":69,"Weight":1025},index=[1])


# In[151]:


new_data


# In[154]:


final_ml_cc.predict(new_data)


# In[155]:


final_ml_cc.predict(toyo5)#automatic prediction of price with 87% accuracy.


# In[ ]:




