#!/usr/bin/env python
# coding: utf-8

# # 股票数据整理（6.7）

# In[56]:


# conda install rpy2


# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import rpy2
import rpy2.robjects as robjects
from sklearn.utils import resample


# dt1 = pd.read_csv('股票数据/行情序列2001年周数据.csv')
# dt2 = pd.read_csv('股票数据/行情序列2002年周数据.csv')
# dt3 = pd.read_csv('股票数据/行情序列2003年周数据.csv')
# dt4 = pd.read_csv('股票数据/行情序列2004-5年周数据.csv')
# dt5 = pd.read_csv('股票数据/行情序列2006-2009年周数据.csv')
# dt6 = pd.read_csv('股票数据/行情序列2010-2013年周数据.csv')
# dt7 = pd.read_csv('股票数据/行情序列2014-2018年周数据.csv')

# dt = pd.concat([dt1,dt2,dt3,dt4,dt5,dt6,dt7], axis = 0, ignore_index = True)

# dt_na = dt.dropna()

# dt_na

# dt.shape

# dt_na.shape

# dt_na.to_csv('数据已合并.csv', index = False, encoding='utf_8_sig')

# # 数据初步描述分析（6.8）
# 
# 见【data description script.R】文件
# 
# ***描述性分析结果*** 
# 1. 一共截取328只基金的累积净值数据，运营时间最长的基金从周2005/1/7 - 周2018/12/28，包含715个数据点，有85%的基金单支拥有50 ~ 400个数据点, 其中有25.8%的基金单支拥有200~250个数据点。
# 2. 由04年至18年，每年数据点数量由最初的1个（仅2004/12/31, 上证500ETF一支）至15674个逐年增多。
# 3. 由04年至18年，基金数量由最初的1支至328支逐年增多，13年基金数量超过100支，14年后基金数量猛增。
# 

# # 数据在模型Kosowski、Fama&French的抽样试验（6.9 - 6.15）

# In[2]:


# 基金数据导入
funds = pd.read_csv('funds_data.csv')


# In[3]:


funds.shape


# In[4]:


funds.head(5)


# In[5]:


funds.info()


# In[6]:


# 基金回报率计算
funds['单位净值(初)']=funds.groupby('代码')['周单位净值(元)'].shift(1)
funds['净回报率'] = funds['周单位净值(元)']/funds['单位净值(初)'] - 1


# In[7]:


funds.head(10)


# In[8]:


# Cahart模型因子数据导入
cahart = pd.read_csv('fivefactor_weekly.csv')


# In[9]:


cahart.shape


# In[10]:


cahart.head(5)


# In[11]:


cahart.info()


# In[12]:


# 时间类数据格式调整
funds['时间'] = pd.to_datetime(funds['时间'], format='%Y-%m-%d')
cahart['trdwk'] = pd.to_datetime(cahart['trdwk'], format='%Y-%m-%d')


# ## Kosowaski模型（6.9 - 6.11）
# 
# ### 1、构建残差population
# 
# 按照Kosowaski模型，根据Cahart四因子模型构建各基金alpha, 周因子数据由国泰安数据库提供。
# 
# ##### 对齐四因子与基金周收益时间点
# 取四因子及最长运营的基金周收益05/01/07 - 18/12/28数据进行right_join, 结果如下；

# ***观察结果***：周因子数据与单支基金周净值数据存在差值，周因子数据缺失2006/10/6、2007/10/5、2009/12/31、2013/12/31四周数据，为计算alpha、载荷及残差用缺失值上值进行填补。
# ![image.png](attachment:image.png)

# #### 合并四因子与基金收益率

# In[13]:


# 取Cahart模型所需因子：市场风险因子（mkt_rf）、规模风险因子（smb）、账面市值比风险因子（hml）、惯性/动量因子（umd）及无风险利率（rf）
c1 = cahart[(cahart['trdwk']>="2005-01-07")&(cahart['trdwk']<="2018-12-28")]
c2 = c1[['trdwk','mkt_rf','smb','hml','umd','rf']]
# 取基金净值有效时间段内数据
f = funds[(funds['时间']>='2005-01-07')&(funds['时间']<='2018-12-28')]
f = f.rename(columns = {'时间': 'trdwk'})
# 合并因子与基金净值
cmodel = pd.merge(f, c2, how='left', on='trdwk')


# In[14]:


# 处理缺失值： 用上一个非缺失值填补
cmodel = cmodel.fillna(method='ffill')


# In[15]:


# 调整合并后数据列名
cmodel = cmodel.drop(columns = ['周单位净值(元)', '单位净值(初)'])
cmodel = cmodel.rename(columns = {'代码': 'code', '简称': 'name', '净回报率': 'ri'})


# In[16]:


cmodel.shape


# In[17]:


cmodel.head(5)


# In[18]:


# 查看净值与四因子之间相关性
corr_dt = cmodel[3:]


# In[19]:


corr_dt.corr()


# #### 建立OLS回归模型
# 
# 根据论文描述，Kosowaski模型是对单支基金先OLS回归得到残差，再对单支基金的残差pop进行bootstrap抽样，一次bootstrap抽样能得到一组残差样本，在计算出一组伪净值数据，根据一组伪净值数据用OLS回归计算出一个伪alpha。
# 
# bootstrap对单支基金抽样b次能得到b个伪alpha，抽样n个基金能构成一个n乘以b的alpha分布。

# In[22]:


funds_list = cmodel.code.unique() # 基金清单


# In[23]:


# 多基金残差计算
def res_caculator(f):
    Y = f['ri']-f['rf']
    result = sm.OLS(Y, sm.add_constant(f.loc[:, ['mkt_rf', 'smb', 'hml', 'umd']])).fit() # 净值根据Cahart四因子做线性回归
    y_dot = result.fittedvalues # 线性方程拟合后计算净值预测值
    res = Y - y_dot # 净值与净值预测值相减形成残差
    
    # 记录回归方程中各参数取值
    coef = {}
    coef.update(alpha_dot = result.params[0], mkt_b = result.params[1], smb_b = result.params[2], hml_b = result.params[3],  umd_b = result.params[4])
    
    # 返回res计算后的单支基金数据及对应的四因子回归方程系数
    return (res, coef) 


# cmodel.groupby('code').apply(lambda x:)

# In[24]:


########## TEST ###########
funds1 = cmodel[cmodel['code'] == funds_list[0]].reset_index(drop=True) # 使用基金清单中第一支基金测试以上残差计算函数
plt1 = res_caculator(funds1)[0]
coefs = res_caculator(funds1)[1]


# In[25]:


########## TEST ###########
fig = plt.figure()  # 残差计算结果，表现出四因子估计值与基金真是净值时序之间的关系 
ax = fig.add_subplot(111)
funds1.loc[:,'res'] = plt1
ax.plot(funds1.trdwk, funds1.res) # 残差分布在[-0.1, 0.1]之间


# ### 2. 构建基金回报率伪时间序列、伪Alpha计算方程 （6.15）
# 
# Kosowski模型创造基金回报率伪时间序列的方式是首先对残差数列进行有放回抽样，使用抽样残差数列，配合以下公式计算基金回报率伪时间序列：
# ![image.png](attachment:image.png)
# 
# #### 残差抽样方程

# In[40]:


# kosowski模型针对残差的Bootstrap抽样（******* 抽样部分之后要加入最优时序分块和家族分块步骤 ********）
def k_sampling(res):
    # 残余抽样
    n = len(res)
    spl_res = resample(res, n_samples = n, replace = 1)
    
    return spl_res


# In[46]:


def k_sampling1(res):
    # 最优时序分块采样
    spl_res = robjects.r['tsbootstrap'](res, nb = 500, statistic = mean, m = 1)
    return spl_res


# In[27]:


########## TEST ###########
plt1_k_sample = k_sampling(plt1)
plt1_k_sample.shape


# In[48]:


k_sampling1(plt1)


# #### 伪Alpha计算方程
# 得到回报率伪时间序列后，再通过CahartOLS再次回归，得到该基金伪Alpha

# In[28]:


# kosowski模型抽样残差进行伪回报及伪alpha计算
def k_fake_alpha_calculator(res, coef, cahart_factors):
    res = res.reset_index(drop=True)
    
    # 通过残差抽样计算伪回报率
    fake_y = cahart_factors['mkt_rf']*coef.get('mkt_b') + cahart_factors['smb']*coef.get('smb_b') +                    cahart_factors['hml']*coef.get('hml_b') + cahart_factors['umd']*coef.get('umd_b') + res
    
    # 得到伪回报率，通过伪回报率Cahart OLS回归得到基金伪Alpha.
    fake_result = sm.OLS(fake_y, sm.add_constant(cahart_factors.loc[:, 'mkt_rf':'umd'])).fit()
    fake_alpha = fake_result.params[0]
    fake_ta = fake_result.tvalues[0]
    
    return (fake_alpha, fake_ta)


# In[29]:


########## TEST ###########
k_fake_alpha_calculator(plt1_k_sample, coefs, funds1)


# ### 3. 计算运气Alpha分布 （6.15 - 6.16）
# 
# ！**目前思路**
# 1. 计算所有基金真实alpha【alpha】及其alpha的t-test值【ta】，对真实alpha进行排序；
# 2. 计算一次所有基金伪alpha【fk_alpha】及其对应的t-test值，并对fk_alpha排序，记为Series1；
# 3. 重复1000次伪alpha序列计算，得到1000个Series.
# 4. 计算1000个Series每个rank(length：328)上的t-test平均数【mean_tfa】
# 5. 在每个rank上，计算【alpha】在1000个【fk_alpha】构成的【运气alpha分布】上小于其的百分比，记为【%<Act】

# In[30]:


cmodel.head(10)


# In[31]:


# 第一步：计算所有基金真实Alpha并排序 #
alpha = []
ta = []

for fund in funds_list:
    tmp_f = cmodel[cmodel['code'] == fund].reset_index(drop=True)
    X = tmp_f.loc[:, 'mkt_rf':'umd']
    Y = tmp_f.ri - tmp_f.rf
    
    true_ols = sm.OLS(Y, sm.add_constant(X)).fit()
    alpha_i = true_ols.params[0]
    ta_i = true_ols.tvalues[0]
    
    alpha.append(alpha_i)
    ta.append(ta_i) 


# In[32]:


# alpha及其t-value数列的排序及整理方程
def series_manipulator(alpha, ta):
    # 将两个List合并到一起，按alpha排序
    alpha = np.array(alpha).reshape(len(alpha),1)
    ta = np.array(ta).reshape(len(ta),1)

    one_fund_array = np.hstack([alpha, ta])
    one_fund_array = one_fund_array[np.argsort(one_fund_array[:,0]),:]
    
    # 将排好序的alpha和其t-value分开，变成两个二维数组
    one_fund_list = np.hsplit(one_fund_array,2)
    
    return one_fund_list


# In[33]:


# 排序
true_alpha_series = series_manipulator(alpha, ta)


# # ! **算力不足问题**
# 目前伪Alpha计算并排序的步骤循环5次大约需要1分钟，20次大约需要4分半，预计如需循环计算1000次需要3~4个小时

# In[42]:


# 第三步：重复计算所有基金伪Alpha series并排序 #
myalphalist = []
mytalist = []

for b in range(20):
    fk_alpha = []
    fk_ta = [] 
    
    # 第二步：计算所有基金伪Alpha并排序，记为Series1 #
    for fund in funds_list:
        tmp_f = cmodel[cmodel['code'] == fund].reset_index(drop=True)
        tmp_cahart = tmp_f.loc[:, 'mkt_rf':'umd']   # 当前基金对应的Cahart四因子数据
        
        tmp_res = res_caculator(tmp_f) # 计算单支基金四因子回归模型的残余
        coef = tmp_res[1] # 当前基金Cahart OLS回归后各因子参数

        #  当前基金残余Bootstrap抽样
        bootstrap_res = k_sampling(tmp_res[0])
        
        #  计算单次抽样res产生的伪净值序列
        fk_data = k_fake_alpha_calculator(bootstrap_res, coef, tmp_cahart) 
        fk_alpha_i = fk_data[0]
        fk_ta_i = fk_data[1]
    
        fk_alpha.append(fk_alpha_i)
        fk_ta.append(fk_ta_i)
   
    one_fund_list = series_manipulator(fk_alpha, fk_ta)
    alpha_series_i = one_fund_list[0]
    ta_series_i = one_fund_list[1]
    
    myalphalist.append(alpha_series_i)
    mytalist.append(ta_series_i)


# In[44]:


mat = np.array(myalphalist)
mat.shape


# In[45]:


mat[19,:,:]


# In[ ]:




