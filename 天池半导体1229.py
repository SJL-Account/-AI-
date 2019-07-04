
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing


# # 函数定义部分

# 获取以列为索引的值为每列空值数量的并且按照空值数量排序的df

# In[2]:

def miss_col(train_df):
    col_miss_df=train_df.isnull().sum(axis=0).reset_index()
    col_miss_df.columns=['col','miss_count']
    col_miss_df=col_miss_df.sort_values(by='miss_count')
    return col_miss_df


# 根据每行数据的类型选出都是浮点型数据的列float_col

# In[3]:

def obtain_x(train_df,xtype):
    xtype_df=train_df.dtypes.reset_index()
    xtype_df.columns=['col','xtype']
    return xtype_df[xtype_df.xtype==xtype].col.values


# 获取日期的列

# In[4]:

def obain_date_col(train_df,float64_col):
    float_date_col=[]
    for col in float64_col:
        if train_df[col].min()>1e13:
            float_date_col.append(col)
    return float_date_col
            


# 找出一行只有一个值得列

# In[5]:

def unique_col(float_df,float64_col):
    unique_col=[]
    for col in float64_col :
        if len(float_df[col].unique())==1:
            unique_col.append(col)
    return unique_col
            


# In[6]:

def cal_corrcoef(float64_df,float64_col,y):
    corr_values=[]
    for col in float64_col:
        #0行1列
        corr_values.append(abs(np.corrcoef(train_df[col].values,y))[0,1])
    corr_pd=pd.DataFrame({'col':float64_col,'corr_values':corr_values})
    return corr_pd.sort_values(by='corr_values',ascending=False)
    


# In[7]:

def fit_model(X,y):
    model=LinearRegression()
    model.fit(X,y)
    return model


# 均方误差

# In[8]:

def msm(y,y_predict):
    n=len(y)
    msm=np.sum((y_predict-y)**2)/n
    return msm


# # 数据挖掘部分

# In[9]:

print 'import train data....'
train_df=pd.read_csv('train.csv')
print 'import train data successfully'


# In[10]:

train_df


# 找出整列都为空值的数据列标Miss_col

# In[11]:

print 'find na data....'
col_miss_df=miss_col(train_df)
col_miss=col_miss_df[col_miss_df.miss_count==500].col
print 'all columns is nan count:',len(col_miss)


# 删除miss_col

# In[12]:

train_df.drop(col_miss,axis=1,inplace=True)
print 'after [clean all columns is nan] shape ',train_df.shape


# In[13]:

print 'obtain float64 data...'
float64_col=obtain_x(train_df,'float64')
print 'float columns count ', len(float64_col)


# 统计他们空值的数量，并选出每列空值小于200的列

# In[14]:

float64_miss_df=train_df[float64_col].isnull().sum(axis=0).reset_index()
float64_miss_df.columns=['col','float_miss_count']
float64_miss_col=float64_miss_df[float64_miss_df.float_miss_count>100].col.values
print 'float miss data count:',len(float64_miss_col)
float64_nonNa_col=[col for col in float64_col if col not in float64_miss_col]
print 'float64 ,na<100,count:',len(float64_nonNa_col)


# 删除日期数据的列

# In[15]:

float_date_col=obain_date_col(train_df,float64_nonNa_col)
float64_nonNa_nonDate_col=[col for col in float64_nonNa_col if col not in float_date_col]
print 'after delete date columns count:',len(float64_col)


# 去除只有一行数据都是一样的列

# In[16]:

unique_col=unique_col(train_df,float64_nonNa_nonDate_col)
print 'only one value has :',len(unique_col)


# In[17]:

float64_nonNa_nonDate_nonUnique_col=[col for col in  float64_nonNa_nonDate_col if col not in unique_col]
print "after clean unique data count :", len(float64_nonNa_nonDate_nonUnique_col)


# 根据上述的数据集用该列的中位数填充其na的值

# In[24]:

new_clear_col=float64_nonNa_nonDate_nonUnique_col
new_clear_df=train_df[new_clear_col]
new_clear_df.fillna(new_clear_df.mean(),inplace=True)


# In[19]:

new_clear_col.remove('Y')
y_train=train_df['Y']
y=y_train.values


# 相关系数大于0.2的列

# In[20]:

corr_df=cal_corrcoef(new_clear_df,new_clear_col,y)


# In[21]:

corr_df=corr_df[corr_df.corr_values>=0.21]
corr_col=corr_df.col.values.tolist()
x_train=new_clear_df[corr_col].values
print 'x_train shape is :',x_train.shape


# In[22]:

corr_df


# 导入测试集

# In[25]:

print 'import test '
test_df=pd.read_csv('test.csv')


# In[26]:

test_df=test_df[corr_col]


# In[27]:

test_df


# In[28]:

test_df=test_df.fillna(test_df.mean(),inplace=True)


# In[29]:

x_test=test_df.values


# 对训练集和测试集和进行标准化

# In[31]:

X=np.vstack((x_train,x_test))
X=preprocessing.scale(X)
x_scale_train=X[: len(x_train)]
x_scale_test=X[len(x_train):]


# 对训练集进行线性回归训练

# In[33]:

model=fit_model(x_scale_train,y)


# # 回归预测部分

# In[ ]:

result=model.predict(x_scale_train[:300])


# In[ ]:

result.shape


# 输出结果

# In[ ]:

subA_df=pd.read_csv('subA.csv',header=None)


# In[ ]:

subA_df['Y']=result


# In[ ]:

subA_df.to_csv('resultA1228.csv',header=None)

