#!/usr/bin/env python
# coding: utf-8

# # Data concatenation
# - Bigger data is right most of the time I guess

# In[ ]:





# In[1]:


with open('psid_controls.txt') as f:
    psid = f.read().splitlines()

with open('psid_controls2.txt') as f:
    psid2 = f.read().splitlines()
    
with open('psid_controls3.txt') as f:
    psid3 = f.read().splitlines()
    
with open('cps_controls.txt') as f:
    cps = f.read().splitlines()
    
with open('cps_controls2.txt') as f:
    cps2 = f.read().splitlines()
    
with open('cps_controls3.txt') as f:
    cps3 = f.read().splitlines()


# In[2]:


def struct(n):
    table = []
    for i in n:
        i = i.split()
        #print(i)
        tmp = []
        for j in i:
            j = float(j)
            tmp.append(j)
        table.append(tmp)
    return table 


# In[3]:


import pandas as pd


# In[6]:


n_cols = """
treatment age education Black Hispanic married nodegree RE74 RE75 RE78

"""
cols= n_cols.split()

data = [psid, psid2, psid3, cps, cps2, cps3]
df = pd.DataFrame()

for i in data:
    df = pd.concat((df, pd.DataFrame(struct(i)))).reset_index(drop=True)
    
df.columns = cols


# In[7]:


nsw = pd.read_csv("nsw_shuffled.csv", index_col = 0)


# In[8]:


nsw


# In[9]:


df


# In[14]:


del df['RE74']


# In[16]:


df.to_csv("psid_cps.csv")


# In[18]:


df = pd.concat((df, nsw)).reset_index(drop=True)


# In[19]:


df = df.sample(frac=1).reset_index(drop=True)


# In[20]:


df


# In[21]:


df.to_csv("nsw_psid_cps.csv")


# In[23]:


df.describe()


# In[24]:


from collections import defaultdict


# In[32]:


df[df.treatment == 1]


# In[33]:


df[df.treatment == 0]


# In[ ]:




