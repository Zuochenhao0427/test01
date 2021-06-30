# 环境设定

# In[5]:

import pandas as pd
import numpy as np
import statsmodels.api as sm
# import matplotlib.pyplot as plt
# import rpy2
# import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
# from sklearn.utils import resample
#from rpy2.robjects import r
from rpy2.robjects import pandas2ri

pandas2ri.activate()

# 数据导入

# In[6]:


# 基金数据导入
funds = pd.read_csv('funds_data.csv')
# Cahart模型因子数据导入
cahart = pd.read_csv('fivefactor_weekly.csv')


# 数据调整

# In[7]:


# 时间类数据格式调整
funds['时间'] = pd.to_datetime(funds['时间'], format='%Y-%m-%d')
cahart['trdwk'] = pd.to_datetime(cahart['trdwk'], format='%Y-%m-%d')


# In[8]:


# 基金回报率计算
funds['单位净值(初)']=funds.groupby('代码')['周单位净值(元)'].shift(1)
funds['净回报率'] = funds['周单位净值(元)']/funds['单位净值(初)'] - 1


# In[9]:


# 取Cahart模型所需因子：市场风险因子（mkt_rf）、规模风险因子（smb）、账面市值比风险因子（hml）、\
#                    惯性/动量因子（umd）及无风险利率（rf）
c1 = cahart[(cahart['trdwk']>="2005-01-07")&(cahart['trdwk']<="2018-12-28")]
c2 = c1[['trdwk','mkt_rf','smb','hml','umd','rf']]
# 取基金净值有效时间段内数据
f = funds[(funds['时间']>='2005-01-07')&(funds['时间']<='2018-12-28')]
f = f.rename(columns = {'时间': 'trdwk'})
# 合并因子与基金净值
cmodel = pd.merge(f, c2, how='left', on='trdwk')


# In[10]:


# 处理缺失值： 用上一个非缺失值填补
cmodel = cmodel.fillna(method='ffill')


# In[11]:


# 调整合并后数据列名
cmodel = cmodel.drop(columns = ['周单位净值(元)', '单位净值(初)'])
cmodel = cmodel.rename(columns = {'代码': 'code', '简称': 'name', '净回报率': 'ri'})


# In[12]:


cmodel.head()


# 计算基金池内各基金真实Alpha的估计值并排序（使用Cahart四因子OLS回归模型）

# In[13]:

fmodel = cmodel.groupby("code").filter(lambda x: (len(x) > 50))
funds_list = fmodel.code.unique() # 基金清单


# In[14]:


alpha = []
ta = []

for fund in funds_list:
    tmp_f = cmodel[cmodel['code'] == fund].reset_index(drop=True)
    X = tmp_f.loc[:, 'mkt_rf':'umd']
    Y = tmp_f.ri - tmp_f.rf
    
    true_ols = sm.OLS(Y, sm.add_constant(X)).fit()
    alpha_i = true_ols.params[0]
    ta_i = true_ols.tvalues[0]
    
    alpha.append(alpha_i) # 每只基金的真实Alpha(使用OLS回归后的估计值)
    ta.append(ta_i) # Alpha估计值的t-value


# In[15]:


print("alpha:", alpha[:5],"\n","ta:",ta[:5])


# ## Fama&French模型（6.18 - 6.24）
# 
# Fama&French模拟回报率伪时间序列的方式与Kosowsk模型略有不同，主要步骤如下；
# 1. 计算所有基金真实alpha的估计值(已计算)
# 2. 使用F&F公式计算单支基金的伪回报率；***伪回报率R`=真实回报率R - 真实alpha的估计值***
# 3. 对基金按月份分组，Bootstrap月数据，模拟伪回报率的时间序列，这种方法保留了周数据中基金组的截面数据
# 4. 对单支基金伪回报率的伪时间序列做Cahart OLS回归，得到单支基金伪alpha
# 5. 排序所有基金的伪alpha

# In[16]:


funds_list
# alpha = np.array(alpha).reshape(328,) # 调整alpha和ta的数列构造
# ta = np.array(ta).reshape(328,)

# 转换成数据表格
df ={'code':funds_list, 'alpha':alpha, 'ta':ta }
df = pd.DataFrame(df)

# 与cahart四因子数据合并
fmodel = pd.merge(cmodel, df, on='code')

# 计算基金的伪回报率
fmodel['fk_r'] = fmodel['ri'] - fmodel['alpha']
fmodel.head(5)


# 对基金按月份分组，Bootstrap月数据
# 1. 取基金代码、数据时间戳、伪回报列，按时间戳列出所有基金时间点上的伪回报（窄表转宽表）

# In[17]:


def long_to_wide_f(fk_r_df):
    # 保留原始时序标签
    time = pd.DataFrame(fk_r_df['trdwk'].unique()).sort_values(by=0).reset_index(drop=True) 

    # 行-时间，列-所有基金，单元格表示一只基金在一个时间点的伪回报率（回报减去alpha）
    f_spl = fk_r_df.pivot_table(index='trdwk',
                                columns = 'code',
                                values = 'fk_r') 

    # 去掉宽表中的多层索引
    f_spl1 = np.array(f_spl)
    columns = f_spl.columns.values
    f_spl_df = pd.DataFrame(f_spl1, columns = columns).fillna(0) # 补足某些基金在早期时间点上的NaN
    
    return (f_spl_df,time)


# In[18]:


fmodel_1 = fmodel[['trdwk','code','fk_r']]
f_spl_df = long_to_wide_f(fmodel_1)


# In[19]:


f_spl_df[0]


# 2. 按时间戳index进行抽样(已将时间戳由日期转为int1-713)

# In[20]:


# 时间戳index抽样1000次（nb = 1000）
def f_sampling(f_spl_df):
    f_series=pd.Series(range(len(f_spl_df[0])))
    tseries = importr('tseries')
    #tsbootstrap = robjects.r['tsbootstrap']()
    spl_f = tseries.tsbootstrap(f_series, nb = 1000, m = 1)
    spl_f0 = np.array(spl_f) # (715,1000)
    spl_f1 = spl_f0.T # 转置为（1000，715）
    
    return spl_f1


# 3. 按每一次抽样时间戳制作伪时间序列，和真实时间轴上的Cahart四因子按按时间索引进行合并

# In[21]:


# 工具1：合并cahart四因子与基金回报率伪时序序列
def f_ols_caculator(df_wide, time, c_factors):
    
    # 宽表加上短时间格式的时间戳（由[0,715]变成['2005-1-7', '2018-12-31']）
    df_wide0 = pd.merge(df_wide,time,left_index = True,right_index=True).rename(columns = {0:'trdwk'})
    
    # 与cahart周频四因子数据合并
    c_factors.reset_index(drop=True)
    df_wide1 = pd.merge(df_wide0, c_factors, how='left', on='trdwk').fillna(method='ffill').set_index('trdwk')

    return df_wide1


# In[22]:


# 工具2：alpha和t-value数列按alpha值排序
def series_manipulator(alpha, ta):
    # 将两个List合并到一起，按alpha排序
    alpha = np.array(alpha).reshape(len(alpha),1)
    ta = np.array(ta).reshape(len(ta),1)

    one_fund_array = np.hstack([alpha, ta])
    one_fund_array = one_fund_array[np.argsort(one_fund_array[:,0]),:]
    
    # 将排好序的alpha和其t-value分开，变成两个二维数组
    one_fund_list = np.hsplit(one_fund_array,2)
    
    return one_fund_list  


# 4. 对单支基金伪回报率的伪时间序列做Cahart OLS回归，得到单支基金伪alpha
# 5. 排序所有基金的伪alpha

# In[20]:


f_fk_alpha_series = []
f_fk_ta_series = []

spl_f1 = f_sampling(f_spl_df)

for i in spl_f1:
    f_spl_df_b = f_spl_df[0].iloc[i].reset_index(drop=True) # 按一次抽样时间戳序列排列各基金伪回报
    f_spl_df_ols = f_ols_caculator(f_spl_df_b,f_spl_df[1],c2) # 工具1
    
    # OLS回归计算伪alpha分布
    f_fk_alpha = []
    f_fk_ta = []
    for fund in funds_list:
        Y = f_spl_df_ols[fund] - f_spl_df_ols['rf']
        X = f_spl_df_ols.loc[:,'mkt_rf':'umd']
        fake_result = sm.OLS(Y, sm.add_constant(X)).fit()
        fk_alpha_i = fake_result.params[0]
        fk_ta_i = fake_result.tvalues[0]
    
        f_fk_alpha.append(fk_alpha_i)
        f_fk_ta.append(fk_ta_i)
    
    f_list = series_manipulator(f_fk_alpha, f_fk_ta) # 工具2
    
    f_fk_alpha_series.append(f_list[0]) 
    f_fk_ta_series.append(f_list[1]) 


# f&f模型制表
# 1. 对真实alpha的t-value排序

# In[23]:


true_funds_dt = series_manipulator(alpha,ta)
t_ta = np.array(true_funds_dt[1]).flatten() # EXP: t_ta.shape = (328,)


# 2. 对伪alpha数组计算t-value在每个alpha level的平均值

# In[ ]:


f_ta_array = np.array(f_fk_ta_series).reshape(len(funds_list),len(spl_f1)) # EXP: f_ta_array.shape = (1000,328)
sim_ta = f_ta_array.mean(axis = 1) # EXP f_ta_mean.shape = (328,)


# 3. 排列每个alpha level上的t-value

# In[ ]:


f_ta_dist = f_ta_array.T # EXP: f_ta_dist.shape = (328,1000)


# In[125]:


# 产出每个alpha level上的%<Act值
perc_act = []
for i in range(len(funds_list)):
    lower = len(f_ta_dist[i][f_ta_dist[i]<ta[i]])/len(f_ta_dist[i])
    perc_act.append(lower)


# In[ ]:


print(len(perc_act), perc_act[:10]) # EXP: len(perc_act) = len(funds_list) = 328, 且从小到大有序排列


# In[24]:


mydict = {'Act': t_ta,
          'Mean.Sim': sim_ta,
          'Perc_less_than_Act': perc_act
         }


# In[25]:


mydict1 = pd.DataFrame(mydict)
mydict1.to_csv('f_frame.csv')
# print('\n', 'mydict1:', '\n', mydict1)


# #### --------------------------------- Split Line ------------------------------------- ####

# ta = np.random.randint(0,100,(328,))
# f_ta_dist=np.random.randint(0,100,(328,1000))
# f_ta_dist[:,1]
# len(ta)
# 
# ta[1]
# 
# len(f_ta_dist[:,1])
# 
# len(f_ta_dist[0])
# 
# len(f_ta_dist[:,1])
# 
# len(f_ta_dist[0])
# 
# len(f_ta_dist[326][f_ta_dist[326]<ta[326]])
# 
# perc_act
# 
# test1 = np.random.random((1000,715))
# len(test1)
# 
# test_list = []
# for i in range(1000):
#     test_list.append(test1)

# np.random.seed(0)
# t1 = np.random.random((3,2,1))
# t1
# 
# t1.reshape(3,2)
# 
# t1.reshape(3,2).mean(axis = 0)
# 
# np.random.seed(0)
# np.random.random((3,2)) # np.random.random((2,3)).T

# In[ ]:




