#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dataset = pd.read_csv("E://user//Documents//Sparks Foundation//student_scores.csv")


# In[4]:


dataset.shape


# In[5]:


dataset.head()


# In[6]:


dataset.describe()


# In[7]:


dataset.plot(x='Hours', y='Scores', kind='bar')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[8]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[16]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[17]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[11]:


print(regressor.intercept_)


# In[12]:


print(regressor.coef_)


# In[18]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[21]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[15]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:




