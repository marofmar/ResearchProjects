#!/usr/bin/env python
# coding: utf-8

# ## New variables up to date (done)
# ## Matching graphs exported as png files (done)
# ## Make Balance Tables (done)
# ## Make Post Tabbles (---ing)
# ## Added codes to visualize Random Forest (---ing)
# ## Build Class which contain method functions (---ing)

# # New variables, new matching
# - Gender specific

# In[1]:


import numpy as np
import pandas as pd
from collections import Counter
import copy
import random
import matplotlib.pyplot as plt 
import math 
import random
from datetime import date


# In[2]:


from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error


# In[3]:


#dt_label = pd.read_csv("./dataset/pre_post_main_data_final_label.csv")
dt = pd.read_csv("./dataset/pre_post_main_data_final_update1114.csv")
# data updated Nov 14 


# In[4]:


#dt_label.shape


# In[5]:


dt.shape


# In[6]:


#dt_label.head()


# In[7]:


dt.head()


# In[8]:


#dt_label.iloc[:5, :]


# In[9]:


dt.iloc[:5, :]


# In[10]:


# what is the diff btw dt and dt_label? 
list(dt.columns)


# In[11]:


# what is the diff btw dt and dt_label? 
#list(dt_label.columns)


# In[12]:


# for i in list(dt.columns):
#     if 'STEM' in i:
#         print(i)


# In[13]:


# dt_label.filter(like='STEM').sample(10)


# In[14]:


# dt.filter(like='STEM').sample(10)


# In[15]:


# a = dt[['STEM1','STEM']]


# In[16]:


# a['sum']  = dt['STEM1']+dt['STEM']


# In[17]:


# Counter(a['sum'])


# In[18]:


# a.sample(3)


# In[19]:


# a[a['sum'].isnull()]


# In[20]:


# a[a['sum']==-1.4265606000000002]


# In[21]:


# # gender 에 따라 나누기 
# dt.gender.unique()


# ## STEM1 is the response variable

# In[22]:


# Counter(dt.gender)


# In[23]:


# gender 변수에 nan 값도 있고 12 로 입력된 잘못된 값도 있음
# 잘못 입력된 혹은 입력되지 않은 값 3개 제외하면 남-여 합쳐서 828개 데이터

# zero = dt[dt.gender == 0] girls `
# one = dt[dt.gender ==1] 


# In[24]:


for i in dt.columns:
    if 'missing' in i:
        print(i)


# In[ ]:





# In[25]:


cols = """
age Female Standard7 Form1 Form2 
Christianity 
fath_live mom_live fath_edu_years mom_edu_years piped_water energy_main flush_toilet room_sleep 
comp_school_1 comp_school_2 hour_mathsci 
Extroversion Agreeableness Conscientiousness Neuroticism Openness 
edu_goal4 edu_goal5 edu_goal6 marri_age_male marri_age_female 
occupation2 occupation4 
occupation5 occupation6 occupation8 occupation9 occupation_group 
gender_pca esteem_pca 
comp_school_missing fath_live_missing mom_live_missing fath_edu_years_missing mom_edu_years_missing 
personality_missing gender_pca_missing
STEM1
"""

# gender 에 따라서 따로 매치을 하므로 Female 변수를 빼고 매칭을 진행해 보았습니다. - > 나중에 그냥 Female 칼럼을 삭제하는 것으로 바꾸었습니다
# cols_wo_Female = """
# age Standard7 Form1 Form2 
# Christianity 
# fath_live mom_live fath_edu_years mom_edu_years piped_water energy_main flush_toilet room_sleep 
# comp_school_1 comp_school_2 hour_mathsci 
# Extroversion Agreeableness Conscientiousness Neuroticism Openness 
# edu_goal4 edu_goal5 edu_goal6 marri_age_male marri_age_female 
# occupation2 occupation4 
# occupation5 occupation6 occupation8 occupation9 occupation_group 
# gender_pca esteem_pca 
# comp_school_missing fath_live_missing mom_live_missing fath_edu_years_missing mom_edu_years_missing 
# personality_missing gender_pca_missing
# STEM1
# """


# In[26]:


cols.split()


# In[27]:


PRE = cols.split()
#PREgender = cols_wo_Female.split()


# In[28]:


GB = dt[PRE]


# In[29]:


# dt[dt.gender == 0] 
G = GB[GB.Female == 0]# Female value 0 for girls 
B = GB[GB.Female == 1]

# G: girls
# B: boys


# In[ ]:





# In[30]:


GB.shape


# In[31]:


print(G.shape, B.shape)


# In[32]:


GB.isnull().sum()


# In[33]:


GB[GB.STEM1.isnull()]


# In[34]:


GB[GB.gender_pca.isnull()]


# In[35]:


GB[GB.esteem_pca.isnull()]


# In[36]:


GB[GB.age == 0.0]


# In[37]:


GB = GB.drop([336, 337, 338])
dt = dt.drop([336, 337, 338])
GB.isnull().sum()


# In[38]:


G = GB[GB.Female == 0] # girls
B = GB[GB.Female == 1] # boys 


# In[39]:


B.isnull().sum() # gender and esteem pca 3 null in Boys


# In[40]:


G.isnull().sum() # same here in girls


# In[41]:


print(GB.shape, G.shape, B.shape)


# In[42]:


del G['Female'] # 성별 매칭에서는 Female 변수 제외 
del B['Female']


# In[43]:


print(GB.shape, G.shape, B.shape)


# In[44]:


logit = LogisticRegression(
            penalty = 'l2', #l1, l2, leasticnet, none default: l2
            fit_intercept = True, 
            intercept_scaling = 1, #float
            random_state= 43, 
            solver = 'liblinear', # newton-cg, lbfgs, liblinear, sag, saga default: lbfgs 
            max_iter = 1000, 
            multi_class = 'ovr',
            warm_start = False)

lasso = Lasso(alpha=1.0,
            fit_intercept=False, # not centered around zero 
            normalize=True, # subtract mean and divide with l2 norm
            precompute=False, # usd Gram matrix, always True when sparse input 
            copy_X=True, # copy X and let X be stayed as it is 
            max_iter=1000, 
            tol=0.001, # the tolerance for the optimization 
            warm_start=False, # set True; reuse the solutoin of the prev call to fit
            positive=False, 
            random_state= 43,  # random_state seed 
            selection='cyclic') # cyclic, random: faster if tol is higher than 1e-4

ridge = Ridge(
            alpha = 1.0, 
            fit_intercept = True,
            random_state= 43, 
            solver = 'auto' )

rf = RandomForestClassifier(n_estimators = 100, 
                            random_state= 43, 
                            max_depth = 5)


# In[45]:


# logit = LogisticRegression()
# lasso = Lasso()
# ridge = Ridge()
# rf = RandomForestClassifier()


# # 1. Girls and Boys 

# In[46]:


# features = copy.deepcopy(GB)
# treatment = features.pop('STEM1')

# logit.fit(features, treatment)


# In[47]:


def pre_match(data, treatment_colname, matching_algorithm):
    features = copy.deepcopy(data)
    treatment = features.pop(treatment_colname)
    matching_algorithm.fit(features, treatment) 
    if matching_algorithm == logit:
        data[str(matching_algorithm)[:5]] = matching_algorithm.predict(features) 
        data[str(matching_algorithm)[:5]] = matching_algorithm.predict_proba(features) 
    elif matching_algorithm == rf:
        data[str(matching_algorithm)[:5]] = matching_algorithm.predict(features) 
        data[str(matching_algorithm)[:5]] = matching_algorithm.predict_proba(features) 
    else:
        data[str(matching_algorithm)[:5]] = matching_algorithm.predict(features) 
    
    return data 


# In[48]:


pre_match(GB, 'STEM1', logit)
pre_match(GB, 'STEM1', lasso)
pre_match(GB, 'STEM1', ridge)
pre_match(GB, 'STEM1', rf)


# In[49]:


#GB.to_csv("1109_GirlsBoys_prematching")


# In[50]:


def show_pre_match_before(data, treatment_colname, algo_colname):
    plt.figure(figsize = (9, 6))
    
    treatment = data[data[treatment_colname]==1]
    control = data[data[treatment_colname]==0]
    
    dic_treat = dict(list(zip(treatment[algo_colname].index, treatment[algo_colname])))
    dic_control = dict(list(zip(control[algo_colname].index, control[algo_colname])))
    
    plt.title(algo_colname)
    plt.hist(list(dic_treat.values()), density = True, bins = 50, alpha = 0.5, label = 'STEM')
    plt.hist(list(dic_control.values()), density = True, bins = 50, alpha = 0.5, label = 'Non_STEM')
    plt.legend(loc = 'upper right')
    
    plt.savefig('pre_matching_before '+algo_colname+str(date.today())+'.png')
    plt.show()
    
    


# In[51]:


show_pre_match_before(GB, 'STEM1', 'Logis')


# In[52]:


show_pre_match_before(GB, 'STEM1', 'Lasso')


# In[53]:


show_pre_match_before(GB, 'STEM1', 'Ridge')


# In[54]:


show_pre_match_before(GB, 'STEM1', 'Rando')


# In[55]:


def dist(a,b):
    """
    목적: 절댓값 거리 구하기
    
    인풋
    a: 어떤 실수값 1
    b: 어떤 실수값 2
    
    리턴
    두 실수 a,b 사이의 절대값 차이 
    
    """
    return abs(a-b) 

# return list of pairs (treat, control)
def match_sample_without_replace(d_trt, d_ctrl):
    """
    목적: replace 없이 샘플링을 하는 함수
    
    """
    sam = []
    for k, v in d_trt.items():
        tmp = {}
        for k_, v_ in d_ctrl.items():
            tmp[k_] = dist(v, v_) 
        #print(k, tmp)
        mini = min(tmp, key = tmp.get) 
        sam.append((k, mini)) 
        #print(mini) 
        del d_ctrl[mini]
    return sam

def match_sample_with_replace(d_trt, d_ctrl):
    """
    목적: with replacement로 샘플링을 하는 함수 
    
    인풋
    d_trt: treatment 값이 1인 데이터 인덱스와 특정 regressor에 따른 점수를 묶어 나타낸 dictionary 형식의 데이터
    d_ctrl: treatment 값이 0인 데이터 인덱스와 특정 regressor에 따른 점수를 묶어 나타낸 dictionary 형식의 데이터
    
    리턴
    treatment와 control 점수 차이가 최소인 한 쌍의 인덱스를 뽑아서, 한 쪽은 treatment sample로, 다른쪽은 control sample로 들어갑니다. 
    
    1. input: dictionary data type, treatment, control 
    2. first input is the actor, and second is the responder 
    3. return the list of indices for each group: treatment and control 5
    """
    sam = []
    for k, v in d_trt.items():
        tmp = {}
        for k_, v_ in d_ctrl.items():
            tmp[k_] = dist(v, v_)
        mini = min(tmp, key = tmp.get)
        sam.append((k, mini))
    treat = []
    control = []
    for t,c in sam:
        treat.append(t)
        control.append(c) 
        
    treat = list(set(treat)) 
    control = list(set(control))
    # indices
    
    return treat, control


# In[56]:


def zip_up_idx_reg(data, treatment_colname, reg_col_name):
    """
    목적: regressor별로 treatment 데이터의 인덱스와 regresson 값, control 데이터의 인덱스와 regression 값을 를 
        dictionary 형식으로 묶어내기
    
    인풋
    treat: treatment 값이 1인 데이터프레임
    control: treatment 값이 0인 데이터프레임
    regf_col_name: 사용할 regressor의 칼럼 이름
    
    리턴
    treatment가 1인 데이터의 인덱스값과 해당 regressor의 값 
    treatment가 0인 데이터의 인덱스값과 해당 regressor의 값 
    
    treat: dataframe treatment value is 1
    control: dataframe treatment value is 0 
    reg_col_name: column name of a specific target regressor [Logit, Lasso, Ridge, RF] Sep 27 2020 
    
    reeturn the dictionary key: value pair (key: index of the data, value: regressor value)
    """
    
    treat = data[data[treatment_colname]==1]
    control = data[data[treatment_colname]==0]
    
    ones = dict(list(zip(treat[reg_col_name].index, treat[reg_col_name])))
    zeros = dict(list(zip(control[reg_col_name].index, control[reg_col_name])))
    return ones, zeros


# In[57]:


lasso_one, lasso_zero = zip_up_idx_reg(GB, 'STEM1', 'Lasso')
logit_one, logit_zero = zip_up_idx_reg(GB, 'STEM1', 'Logis')
ridge_one, ridge_zero = zip_up_idx_reg(GB, 'STEM1', 'Ridge')
rf_one, rf_zero = zip_up_idx_reg(GB, 'STEM1', 'Rando')


# In[58]:


logit_treat, logit_control = match_sample_with_replace(logit_one, logit_zero)
lasso_treat, lasso_control = match_sample_with_replace(lasso_one, lasso_zero) # return the inndices, 인덱스 모인 리스트
ridge_treat, ridge_control = match_sample_with_replace(ridge_one, ridge_zero)
rf_treat, rf_control = match_sample_with_replace(rf_one, rf_zero)


# In[59]:


def save_pre_match_after(data, treatment_colname, algo_colname):
    
    treatment = data[data[treatment_colname]==1]
    control = data[data[treatment_colname]==0]
    
    dic_treat = dict(list(zip(treatment[algo_colname].index, treatment[algo_colname])))
    dic_control = dict(list(zip(control[algo_colname].index, control[algo_colname])))
    
    matched_treat, matched_control = match_sample_with_replace(dic_treat, dic_control)
    # indices
    
    treat_data = data.loc[matched_treat]
    control_data = data.loc[matched_control]
    print(algo_colname,"\ntreated: ", treat_data.shape, "\ncontrolled: ", control_data.shape,"\n")
    return treat_data, control_data


# In[60]:


logit_stem, logit_non = save_pre_match_after(GB, 'STEM1', 'Logis')
# 48은 변수 44개에 4 개의 regressor 별 propensity score columns 를 포함한 것  


# In[61]:


lasso_stem, lasso_non = save_pre_match_after(GB, 'STEM1', 'Lasso')
ridge_stem, ridge_non = save_pre_match_after(GB, 'STEM1', 'Ridge')
rf_stem, rf_non = save_pre_match_after(GB, 'STEM1', 'Rando')


# # Check

# In[62]:


def show_pre_match_after(data, treatment_colname, algo_colname):
    plt.figure(figsize = (9, 6))
    
    treatment = data[data[treatment_colname]==1]
    control = data[data[treatment_colname]==0]
    
    dic_treat = dict(list(zip(treatment[algo_colname].index, treatment[algo_colname])))
    dic_control = dict(list(zip(control[algo_colname].index, control[algo_colname])))
    
    matched_treat, matched_control = match_sample_with_replace(dic_treat, dic_control)
    # indices
    
#     treat_data = data.loc[matched_treat]
#     control_data = data.loc[matched_control]
    
    plt.title(algo_colname)
    plt.hist([dic_treat[x] for x in matched_treat], density = True, bins = 50, alpha = 0.5, color = 'r', label = 'STEM')
    plt.hist([dic_control[x] for x in matched_control], density = True, bins = 50, alpha = 0.5, color = 'c', label = 'Non_STEM')
    plt.legend(loc = 'upper right')
    
    plt.savefig('pre_matching_after '+algo_colname+str(date.today())+'.png')
    plt.show()
    
    


# In[63]:


show_pre_match_after(GB, 'STEM1', 'Logis')


# In[64]:


show_pre_match_after(GB, 'STEM1', 'Lasso')


# In[65]:


show_pre_match_after(GB, 'STEM1', 'Ridge')


# In[66]:


show_pre_match_after(GB, 'STEM1', 'Rando')


# ## Balance Table 만들기 
# - Balance Table 에서 보여줄 변수 칼럼 재정비
# - Stats 정확히! 

# In[67]:


# balance table column names 
table_vars ="""age Female Standard7 Form1 Form2 Form3 
Christianity Islam
fath_live mom_live fath_edu_years mom_edu_years piped_water energy_main flush_toilet room_sleep
comp_school_1 comp_school_2 hour_mathsci
Extroversion Agreeableness Conscientiousness Neuroticism Openness
edu_goal4 edu_goal5 edu_goal6 marri_age
occupation2 occupation4 
occupation5 occupation6 occupation8 occupation9 occupation_group
gender_pca esteem_pca"""

# Form4 Form5 Form6 VE College 제외! 

#marri_age_male marri_age_female
# marri age male , marri age female 을 marri age 로 대체 

table_vars = list(table_vars.split())


# In[68]:


table_vars


# In[69]:


PRE


# In[70]:


print(len(table_vars), len(PRE))


# In[71]:


for i in table_vars:
    if i not in PRE:
        print(i)


# In[72]:


for i in PRE:
    if i not in table_vars:
        print(i)


# In[73]:


GB_ = pd.concat([dt[table_vars], GB.iloc[:, -5:]], axis = 1)


# In[74]:


GB_.columns


# In[75]:


# GB_[['marri_age_male', 'marri_age_female']]


# In[76]:


logit_balance_stem, logit_balance_non = GB_.loc[logit_treat], GB_.loc[logit_control]


# In[77]:


logit_balance_stem.shape


# In[78]:


logit_balance_non.shape


# In[79]:


logit_balance_stem.sample(3)


# In[ ]:





# In[80]:


lasso_balance_stem, lasso_balance_non = GB_.loc[lasso_treat], GB_.loc[lasso_control]
ridge_balance_stem, ridge_balance_non = GB_.loc[ridge_treat], GB_.loc[ridge_control]
rf_balance_stem, rf_balance_non = GB_.loc[rf_treat], GB_.loc[rf_control]
# data frame


# In[81]:


lasso_balance_stem.sample(3)


# In[82]:


def select_the_columns_only(df, list_of_columns):
    return df[list_of_columns]
    


# In[83]:


logit_stem_df = select_the_columns_only(logit_balance_stem, table_vars)
logit_nons_df = select_the_columns_only(logit_balance_non, table_vars)

lasso_stem_df = select_the_columns_only(lasso_balance_stem, table_vars)
lasso_nons_df = select_the_columns_only(lasso_balance_non, table_vars)

ridge_stem_df = select_the_columns_only(ridge_balance_stem, table_vars)
ridge_nons_df = select_the_columns_only(ridge_balance_non, table_vars)

rf_stem_df = select_the_columns_only(rf_balance_stem, table_vars)
rf_nons_df = select_the_columns_only(rf_balance_non, table_vars)

# matched dataframes by algorithms


# In[84]:


rf_stem_df.sample(3)


# In[ ]:





# In[85]:


def join_stats_raw(data, treatment_colname, matching_algorithm=None):
    """
    목적: treametn와 control 그룹을 표로 정리하기
    
    인풋
    treat: treatment 값이 1인 데이터 그룹
    control: treatment 값이 0인 데이터 그룹
    
    리턴
    treatment, control 각각의 갯수(count), 평균, 표준편차를 나타낸 6칼럼의 표 
    
    input
    - treat: treatment dataframe 
    - control: control data sample dataframe 
    
    return
    - treatment data: count, mean, std
    - control sample data: count, mean, std 
    
    """
    treat = data[data[treatment_colname]==1]
    control = data[data[treatment_colname]==0]
    
    STEM = treat.describe().T 
    STEM['count'] = STEM['count'].apply(int) # count datatype supposed to be int not float 
    
    NON_STEM = control.describe().T
    NON_STEM['count']= NON_STEM['count'].apply(int)
    
    STEM = STEM.rename(columns = {'count':'ST_N', 'mean':'ST_Mean', 'std':'ST_SD'}) # change the column name 
    NON_STEM = NON_STEM.rename(columns = {'count':'NS_N', 'mean':'NS_Mean', 'std':'NS_SD'})
    table = pd.concat([STEM.iloc[:, :3], NON_STEM.iloc[:, :3]], axis = 1)
    
    return table
#GB_matched.isnull().sum()


# In[86]:


Counter(dt['STEM1'])


# In[87]:


raw_ = join_stats_raw(GB_, 'STEM1')


# In[88]:


raw_


# In[89]:


def join_stats_treat(treat, control):
    """
    목적: treametn와 control 그룹을 표로 정리하기
    
    인풋
    treat: treatment 값이 1인 데이터 그룹
    control: treatment 값이 0인 데이터 그룹
    
    리턴
    treatment, control 각각의 갯수(count), 평균, 표준편차를 나타낸 6칼럼의 표 
    
    input
    - treat: treatment dataframe 
    - control: control data sample dataframe 
    
    return
    - treatment data: count, mean, std
    - control sample data: count, mean, std 
    
    """
    STEM = treat.describe().T 
    STEM['count'] = STEM['count'].apply(int) # count datatype supposed to be int not float 
    
    NON_STEM = control.describe().T
    NON_STEM['count']= NON_STEM['count'].apply(int)
    
    STEM = STEM.rename(columns = {'count':'ST_N', 'mean':'ST_Mean', 'std':'ST_SD'}) # change the column name 
    NON_STEM = NON_STEM.rename(columns = {'count':'NS_N', 'mean':'NS_Mean', 'std':'NS_SD'})
    table = pd.concat([STEM.iloc[:, :3], NON_STEM.iloc[:, :3]], axis = 1)
    
    return table


# In[90]:


six_logit = join_stats_treat(logit_stem_df, logit_nons_df)
six_lasso = join_stats_treat(lasso_stem_df, lasso_nons_df)
six_ridge = join_stats_treat(ridge_stem_df, ridge_nons_df)
six_rf = join_stats_treat(rf_stem_df, rf_nons_df)


# In[91]:


six_logit


# In[92]:


six_lasso


# In[93]:


six_ridge


# In[94]:


six_rf


# In[95]:


# marri age male, female -> marri age 로 대체되었음 ! 


# In[96]:


def add_diff_mean(df, treat_mean_col_name, control_mean_col_name):
    """
    목적: Difference mean 칼럼을 구합니다.
    
    인풋
    df: 전체 데이터 프레임
    treat_mean_col_name: treatment가 1인 데이터의 mean을 갖는 칼럼 이름
    control_mean_col_name: treatment가 0인 경우
    
    리턴
    Diff_Mean 이라는 새로운 칼럼을 갖는 7축의 테이블
    """
    df['Diff_Mean'] = df[treat_mean_col_name] - df[control_mean_col_name]
    return df


# In[97]:


seven_lasso = add_diff_mean(six_lasso, 'ST_Mean', 'NS_Mean')
seven_logit = add_diff_mean(six_logit, 'ST_Mean', 'NS_Mean')
seven_ridge = add_diff_mean(six_ridge, 'ST_Mean', 'NS_Mean')
seven_rf = add_diff_mean(six_rf, 'ST_Mean', 'NS_Mean')
seven_raw = add_diff_mean(raw_, 'ST_Mean', 'NS_Mean')


# In[98]:


seven_logit


# In[99]:


def add_diff_se(df, treat_sd_col_name, control_sd_col_name):
    """
    input
    - dataframe with 6 basic stats and diff-mean
    - treat_sd_col_name: treatment std column name
    - control_sd_col_name: control sample std column name 
    
    return 
    - dataframe with reordered columns, having diff-mena and diff-sd too 
    """
    treat_N = df.loc['age']['ST_N']
    control_N = df.loc['age']['NS_N']
    df['Diff_SE'] = [math.sqrt(i) for i in df[treat_sd_col_name]**2/int(treat_N) + df[control_sd_col_name]**2/int(control_N)]
    
    # re order the columns
    # mean, std, count, mean, std, count, diff-mean, diff-std
    cols = ['ST_Mean', 'ST_SD', 'ST_N', 'NS_Mean', 'NS_SD', 'NS_N', 'Diff_Mean', 'Diff_SE']
    df = df[cols]
    return df


# In[100]:


eight_lasso = add_diff_se(seven_lasso, 'ST_SD', 'NS_SD')
eight_logit = add_diff_se(seven_logit, 'ST_SD', 'NS_SD')
eight_ridge = add_diff_se(seven_ridge, 'ST_SD', 'NS_SD')
eight_rf = add_diff_se(seven_rf, 'ST_SD', 'NS_SD')
eight_raw = add_diff_se(seven_raw, 'ST_SD', 'NS_SD')


# In[101]:


eight_logit


# In[102]:


eight_raw


# In[103]:


alpha_10 = 1.645 # 90%
alpha_5 = 1.96 # 95% 

def show_dist(df, diff_mean, diff_se, alpha, new_col_name):
    """
    dist consists of pairs (start, end)
    if 0 exists between the start and end, then we cannot reject the null hypothesis that the mean is zero. 
    
    lower limit = diff_mean - (standard error) * (alpha a.k.a. CL coefficient)
    upper limit = diff_mean + (standard error) * (CL coefficient)
    
    """
    dist = []
    
    # this is to verify the diff mean is zero or not
    # H0 : diff mean is zero
    # H1 : diff mean is not zero
    # therefore if this distance range contain zero in it, we cannot reject the null hypothesis H0
    # however, if this dist range is not containing the zero, then we reject the H0, and go for H1...
      
    lower = np.array(df[diff_mean] - df[diff_se]*alpha) # array 
    upper = np.array(df[diff_mean] + df[diff_se]*alpha) # array 
    
    for i in range(len(lower)):
        dist.append((lower[i].round(3), upper[i].round(3)))
    #print(df.shape, len(dist))
    # give new column name? 
    df_new = copy.deepcopy(df)
    df_new[new_col_name] = dist 
    
    return df_new 


# In[104]:


nine_lasso = show_dist(eight_lasso, 'Diff_Mean', 'Diff_SE', alpha_10, '10%')
nine_logit = show_dist(eight_logit, 'Diff_Mean', 'Diff_SE', alpha_10, '10%')
nine_ridge = show_dist(eight_ridge, 'Diff_Mean', 'Diff_SE', alpha_10, '10%')
nine_rf = show_dist(eight_rf, 'Diff_Mean', 'Diff_SE', alpha_10, '10%')
nine_raw = show_dist(eight_raw, 'Diff_Mean', 'Diff_SE', alpha_10, '10%')


# In[105]:


def put_stars(df, diff_mean, conf_int, new_col_name):
    stars = []
    
    for i in range(len(df[conf_int])):
        # i is the idx 
        # (lower, upper) 
        lower = df[conf_int].values[i][0]
        upper = df[conf_int].values[i][-1]
        
        if (lower <= float(0)) and (float(0) <= upper):
            stars.append(' ')
        else:
            stars.append('☆')
    df_new = copy.deepcopy(df)    
    df_new[new_col_name] = stars
    
    return df_new


# In[106]:


ten_lasso = put_stars(nine_lasso, 'Diff_Mean', '10%', '10% stars')
ten_logit = put_stars(nine_logit, 'Diff_Mean', '10%', '10% stars')
ten_ridge = put_stars(nine_ridge, 'Diff_Mean', '10%', '10% stars')
ten_rf = put_stars(nine_rf, 'Diff_Mean', '10%', '10% stars')
ten_raw = put_stars(nine_raw, 'Diff_Mean', '10%', '10% stars')

ele_lasso = show_dist(ten_lasso, 'Diff_Mean', 'Diff_SE', alpha_5, '5%')
ele_logit = show_dist(ten_logit, 'Diff_Mean', 'Diff_SE', alpha_5, '5%')
ele_ridge = show_dist(ten_ridge, 'Diff_Mean', 'Diff_SE', alpha_5, '5%')
ele_rf = show_dist(ten_rf, 'Diff_Mean', 'Diff_SE', alpha_5, '5%')
ele_raw = show_dist(ten_raw, 'Diff_Mean', 'Diff_SE', alpha_5, '5%')

twel_lasso = put_stars(ele_lasso, 'Diff_Mean', '5%', '5% stars')
twel_logit = put_stars(ele_logit, 'Diff_Mean', '5%', '5% stars')
twel_ridge = put_stars(ele_ridge, 'Diff_Mean', '5%', '5% stars')
twel_rf = put_stars(ele_rf, 'Diff_Mean', '5%', '5% stars')
twel_raw = put_stars(ele_raw, 'Diff_Mean', '5%', '5% stars')


# In[107]:


twel_raw


# In[108]:


twel_rf


# In[109]:


twel_logit


# In[110]:


twel_lasso


# In[111]:


twel_ridge.round(3)


# In[112]:


twel_lasso = twel_lasso.round(3)
twel_logit = twel_logit.round(3)
twel_ridge = twel_ridge.round(3)
twel_rf = twel_rf.round(3)
twel_raw = twel_raw.round(3)


# In[113]:


twel_raw


# In[114]:


twel_lasso.to_csv("1116_1110pre_lasso_table.csv")
twel_logit.to_csv("1116_1110pre_logit_table.csv")
twel_ridge.to_csv("1116_1110pre_ridge_table.csv")
twel_rf.to_csv("1116_1110pre_rf_table.csv")
twel_raw.to_csv("1116_1110pre_raw_table.csv")


# In[115]:


# twel_lasso_round.to_csv("1110preROUND_lasso_table.csv")
# twel_logit_round.to_csv("1110preROUND_logit_table.csv")
# twel_ridge_round.to_csv("1110preROUND_ridge_table.csv")
# twel_rf_round.to_csv("1110preROUND_rf_table.csv")
# twel_raw_round.to_csv("1110preROUND_raw_table.csv")


# # 이제 Post variable 추가할 차례
# ## Outcome Table! 

# In[116]:


post_vars = """
fav_sub_mathsci_1 fav_sub_mathsci_2 fav_sub_mathsci_3 hour_KE_post hour_MSC_post hour_GCH_post
edu_goal_post4 edu_goal_post5 edu_goal_post6  dmajor_prefer2 dmajor_prefer5 dmajor_prefer9 dmajor_prefer0 
occupation_post2 occupation_post4 occupation_post5 occupation_post6 occupation_post8 occupation_post9 occupation_post_group 
age_work_post earn25_post earn35_post maxearn35_post minearn35_post esteem_pca_post career_pca 
marri_age_post child_num_post 
gender_pca_post 
know_occ_post  dhear_job_post1 dhear_job_post3 dhear_job_post4 dhear_job_post5 dhear_job_post6 dhear_job_post0
"""

# marri_age_male_post marri_age_female_post 삭제 대신 통합값 넣기 


# In[117]:


for i in dt.columns:
    if 'marri_age' in i:
        print(i)


# In[118]:


POST = post_vars.split()


# In[119]:


len(POST)


# In[120]:


post_table_whole = dt[POST]


# In[121]:


post_table_whole.shape


# In[122]:


logit_post_stem, logit_post_nons = post_table_whole.loc[logit_treat], post_table_whole.loc[logit_control]


# In[123]:


logit_post_stem.shape


# In[124]:


logit_post_nons.shape


# In[125]:


lasso_post_stem, lasso_post_nons = post_table_whole.loc[lasso_treat], post_table_whole.loc[lasso_control]
ridge_post_stem, ridge_post_nons = post_table_whole.loc[ridge_treat], post_table_whole.loc[ridge_control]
rf_post_stem, rf_post_nons = post_table_whole.loc[rf_treat], post_table_whole.loc[rf_control]


# In[126]:


print(lasso_post_stem.shape, lasso_post_nons.shape)


# In[127]:


print(ridge_post_stem.shape, ridge_post_nons.shape)


# In[128]:


print(rf_post_stem.shape, rf_post_nons.shape)


# In[129]:


logit_post_nons.isnull().sum()


# In[130]:


out_six_logit = join_stats_treat(logit_post_stem, logit_post_nons)


# In[131]:


out_six_logit


# In[132]:


#out_six_logit = join_stats_treat(logit_post_stem, logit_post_nons)
out_six_lasso = join_stats_treat(lasso_post_stem, lasso_post_nons)
out_six_ridge = join_stats_treat(ridge_post_stem, ridge_post_nons)
out_six_rf = join_stats_treat(rf_post_stem, rf_post_nons)


# In[133]:


out_six_lasso


# In[134]:


out_six_ridge


# In[135]:


out_six_rf


# In[136]:


out_seven_logit = add_diff_mean(out_six_logit, 'ST_Mean', 'NS_Mean')
out_seven_lasso = add_diff_mean(out_six_lasso, 'ST_Mean', 'NS_Mean')
out_seven_ridge = add_diff_mean(out_six_ridge, 'ST_Mean', 'NS_Mean')
out_seven_rf = add_diff_mean(out_six_rf, 'ST_Mean', 'NS_Mean')


# In[137]:


out_seven_ridge.head()


# In[138]:


def add_diff_se_v2(df):
    """
    row-wise calculation
    standared error = sqrt( (treat_sd**2/treat_n) + (non_sd**2/non_n) )
    """
    
#     treat_N = df.loc['age']['ST_N']
#     control_N = df.loc['age']['NS_N']
    df['Diff_SE'] = [math.sqrt(i) for i in df['ST_SD']**2/df['ST_N'] + df['NS_SD']**2/df['NS_N']]
    
    # re order the columns
    # mean, std, count, mean, std, count, diff-mean, diff-std
    cols = ['ST_Mean', 'ST_SD', 'ST_N', 'NS_Mean', 'NS_SD', 'NS_N', 'Diff_Mean', 'Diff_SE']
    df = df[cols]
    return df


# In[139]:


out_eight_logit = add_diff_se_v2(out_seven_logit)
out_eight_lasso = add_diff_se_v2(out_seven_lasso)
out_eight_ridge = add_diff_se_v2(out_seven_ridge)
out_eight_rf = add_diff_se_v2(out_seven_rf)


# In[140]:


out_eight_rf


# In[141]:


out_eight_logit


# In[142]:


def show_dist_v2(df, new_col_name):
    """
    dist consists of pairs (start, end)
    if 0 exists between the start and end, then we cannot reject the null hypothesis that the mean is zero. 
    
    lower limit = diff_mean - (standard error) * (alpha a.k.a. CL coefficient)
    upper limit = diff_mean + (standard error) * (CL coefficient)
    
    """
    alpha_10 = 1.645 # 90%
    alpha_5 = 1.96 # 95% 
    
    if "10" in new_col_name:
        alpha = alpha_10
    elif "5" in new_col_name:
        alpha = alpha_5
    
    dist = []
    
    # this is to verify the diff mean is zero or not
    # H0 : diff mean is zero
    # H1 : diff mean is not zero
    # therefore if this distance range contain zero in it, we cannot reject the null hypothesis H0
    # however, if this dist range is not containing the zero, then we reject the H0, and go for H1...
      
#     lower = np.array(df['Diff_Mean'] - df['Diff_SE']*alpha) # array 
#     upper = np.array(df['Diff_Mean'] + df['Diff_SE']*alpha) # array 
    for i in df.index:
        lower = np.array(df['Diff_Mean'].loc[i] - df['Diff_SE'].loc[i]*float(alpha)) # array 
        upper = np.array(df['Diff_Mean'].loc[i] + df['Diff_SE'].loc[i]*float(alpha)) # array 

#     for i in range(len(lower)):
#         dist.append((lower[i].round(4), upper[i].round(4)))

        dist.append((lower.round(3), upper.round(3)))
        
    #print(df.shape, len(dist))|
    # give new column name? 
    new_df = copy.deepcopy(df)
    new_df[new_col_name] = dist
    
    return new_df


# In[143]:


out_nine_logit = show_dist_v2(out_eight_logit, '10%')
out_nine_logit


# In[144]:


out_nine_lasso = show_dist_v2(out_eight_lasso,  '10%')
out_nine_ridge = show_dist_v2(out_eight_ridge, '10%')
out_nine_rf = show_dist_v2(out_eight_rf,  '10%')


# In[145]:


out_nine_logit


# In[146]:


dt.index


# In[147]:


def put_stars_v2(df, new_col_name):
    """
    conf_int: 10% or 5% 
    
    """
    
    if '10' in new_col_name:
        conf_int = "10%"
    elif '5' in new_col_name:
        conf_int = "5%"
    
    stars = []
    
    for i in range(len(df[conf_int])):
        # i is the idx 
        # (lower, upper) 
        lower = float(df[conf_int].values[i][0])
        upper = float(df[conf_int].values[i][-1])
        
        if (lower <= float(0)) and (float(0) <= upper):
            stars.append(' ')
        else:
            stars.append('☆')
            
    df_new = copy.deepcopy(df)    
    df_new[new_col_name] = stars
    
    return df_new


# In[148]:


out_ten_logit = put_stars_v2(out_nine_logit,'10% stars')
out_ten_lasso = put_stars_v2(out_nine_lasso,'10% stars')
out_ten_ridge = put_stars_v2(out_nine_ridge,'10% stars')
out_ten_rf = put_stars_v2(out_nine_rf, '10% stars')


# In[149]:


out_ten_rf


# In[150]:


def show_dist_and_put_stars(df, conf_int):
    pass


# In[151]:


out_ele_logit = show_dist_v2(out_ten_logit, '5%')
out_ele_lasso = show_dist_v2(out_ten_lasso, '5%')
out_ele_ridge = show_dist_v2(out_ten_ridge, '5%')
out_ele_rf = show_dist_v2(out_ten_rf, '5%')


# In[152]:


out_twel_logit = put_stars_v2(out_ele_logit, '5% stars')
out_twel_lasso = put_stars_v2(out_ele_lasso, '5% stars')
out_twel_ridge = put_stars_v2(out_ele_ridge, '5% stars')
out_twel_rf = put_stars_v2(out_ele_rf, '5% stars')


# In[153]:


out_twel_logit = out_twel_logit.round(3)
out_twel_lasso = out_twel_lasso.round(3)
out_twel_ridge = out_twel_ridge.round(3)
out_twel_rf = out_twel_rf.round(3)


# In[154]:


out_twel_logit.to_csv("1116post_logit_table.csv")
out_twel_lasso.to_csv("1116post_lasso_table.csv")
out_twel_ridge.to_csv("1116post_ridge_table.csv")
out_twel_rf.to_csv("1116post_rf_table.csv")


# In[155]:


out_twel_logit


# In[156]:


out_twel_lasso


# In[157]:


out_twel_ridge


# In[158]:


out_twel_rf


# In[ ]:




