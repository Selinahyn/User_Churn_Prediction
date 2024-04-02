
# coding: utf-8

# In[2]:



import numpy as np
import pandas as pd
import os


# # 1.数据概览

# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('loss.csv')
df.head()


# In[4]:


df.shape[0]


# In[5]:


# View data variables, total number, missing data, variable measurement(dimension)
def data_overview():
    print("Rows :  " , df.shape[0])
    print("Columns:  " , df.shape[1] )
    print('Missing Value number : ' , df.isnull().sum().values.sum()) #isnull.sum() will sum up each series, so we will also sum up the value once
    print('\nUnique values' , df.nunique())
data_overview()


# In[6]:


df.isnull().sum()


# In[7]:



sns.distplot(df.tenure)


# In[ ]:


sns.distplot(df.MonthlyCharges)


# In[ ]:


df.TotalCharges


# In[ ]:


sns.distplot(df.TotalCharges)


# # 2.数据预处理
# ### 2.1 检查异常值
# ### 第一四分位数-1.5*IQR < 正常值 < 第三四分位数+1.5*IQR
# ### -1,1,1,1,2,2,2,3,3,3,4,4,4

# In[ ]:


#   使用了四分位数范围（IQR）来检测异常值。
#   IQR是第三四分位数（75%分位数）和第一四分位数（25%分位数）之间的差，它是衡量数据分散程度的一种方法。
#   在这个代码中，任何小于第一四分位数减去1.5倍IQR或大于第三四分位数加上1.5倍IQR的值都被认为是可能的异常值。
#   
#   这是一种常用的异常值检测方法，基于的假设是数据应该大致符合正态分布。
def smells(df):
    summary = df.describe(include='all')
    for column in summary.columns:
        if df[column].dtype in ['float64', 'int64']:
            IQR = summary.at['75%', column] - summary.at['25%', column]
            lower_bound = summary.at['25%', column] - 1.5 * IQR
            upper_bound = summary.at['75%', column] + 1.5 * IQR
            if df[column].lt(lower_bound).any() or df[column].gt(upper_bound).any():
                print(f"Column {column} may have outliers.")
        elif df[column].dtype == 'object':
            if df[column].str.len().max() > 255:
                print(f"Column {column} may have strings that are too long.")


# In[ ]:


smells(df)


# ## 结论：SeniorCitizen存在异常值，由于此列只有0 或 1，因此可能是计算误差
# ## 2.2 处理空字符串

# In[ ]:


df.isnull().sum()


# In[ ]:


df= df.replace(' ',np.nan)


# In[ ]:


df.isnull().sum()


# ## 观察异常值所在的账户

# In[ ]:


# 发现11个空值
 
print('LOST：')
print(df.TotalCharges.isnull().sum())


# In[ ]:



df[df.TotalCharges.isnull()]


# In[ ]:



plt.figure(figsize = (12,4))
sns.distplot(df.TotalCharges.notnull().astype(float))


# In[ ]:


sns.distplot(df.TotalCharges)


# ## 2.3 缺失值处理

# In[ ]:


print('清除缺失值')
df = df[df.TotalCharges.notnull()]
df = df.reset_index()
 
df.TotalCharges = df.TotalCharges.astype(float)


# ## 字符串转数字

# In[467]:



df = df.replace({'Yes':1 , 'No' :0})
df.head()


# In[468]:



df = df.replace({'No phone service':0})
df.head()


# ## 2.4 连续型变量处理

# In[469]:


# 转换分箱，以增加对极端值的鲁棒性
print(df.tenure.describe())
# 数据均衡进而采取等宽分箱策略 
 
def tenure_to_bins(series):
    labels = [1,2,3,4,5]
    bins = pd.cut(series , bins = 5 , labels = labels)
    return bins
temp_tenure = df.tenure
df['tenure_group'] = tenure_to_bins(temp_tenure)
df.head()


# In[470]:


#  将数据分开
churn = df[df.Churn == 1]
not_churn = df[df.Churn == 0]
# 类别变数与连续变数分开
Id_col = ['customerID']
target_col = ['Churn']
cat_cols = df.nunique()[df.nunique() < 6].keys().tolist() #取出Series.index，轉成一個list
cat_cols = [col for col in cat_cols if col not in target_col]
num_cols = [x for x in df.columns if x not in Id_col + target_col + cat_cols]


# # 3.EDA(exploratory data analysis）

# In[471]:


# 先導入相關套件
import plotly.offline as py
py.init_notebook_mode(connected=True) #為了能在本地端調用
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff


# ## 3.1 概览目标变量
# #### 下面，使用plt库（一个用于创建交互式图表的Python库）来创建一个饼图，显示了"Churn"列中各个值的数量

# In[472]:


import matplotlib.pyplot as plt
def plot_pie(df, column):
    # 计算各个值的数量
    counts = df[column].value_counts()
    print(counts)
    # 绘制饼图
    plt.pie(counts, labels=counts.index, autopct='%1.3f%%')
    plt.title(f'Pie chart of {column}')
    plt.show()


# In[473]:


plot_pie(df, 'Churn')


# In[474]:


plot_pie(df, 'gender')


# ### 饼图
# def pie_draw(df, column):
#     plt.figure(figsize=(6,6))
#     df.groupby('Churn')[column].value_counts(normalize=True).unstack().plot(kind='pie', subplots=True, autopct='%1.1f%%')
#     plt.title(column + " distribution in Churn")
#     plt.show()

# In[479]:


import matplotlib.pyplot as plt

def draw_pie(df, column):
    # 分别获取流失客户和非流失客户的数据
    churn_df = df[df['Churn'] == 1]
    not_churn_df = df[df['Churn'] == 0]
    # 创建一个新的图形，包含两个子图
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    # 在第一个子图中绘制流失客户的饼图
    axs[0].pie(churn_df[column].value_counts(), labels=churn_df[column].value_counts().index, autopct='%1.1f%%')
    axs[0].set_title('Churn ' + column + ' distribution')
    # 在第二个子图中绘制非流失客户的饼图
    axs[1].pie(not_churn_df[column].value_counts(), labels=not_churn_df[column].value_counts().index, autopct='%1.1f%%')
    axs[1].set_title('Not churn ' + column + ' distribution')

    # 显示图形
    plt.show()


# In[480]:


for col in cat_cols:
    draw_pie(df,col)


# ##### 男性和女性的损失比例相似，说明男性和女性的影响关系不大 
# ##### 在流失的人口中，老年人的比例较高 
# ##### 没有伴侣的人比例更高，可能是因为没有人可以打电话
# ##### 没有孩子的人的损失更高，原因与上面的XDD相同
# #####  是否曾经使用过不相关的电话服务，这表明这不是联系问题
# ##### 不止一个电话服务?总之，使用其他程序的人的流失率是比较高的，相当合理
# ##### 使用No网络服务的人很少丢失，而使用光纤的人应该是相当大的(丢失、不丢失的比例都相当高)，可见网络应该是一个关键因素
# ##### 仅使用一个月(一个月合同)的高流失率说明很多客户可能是所谓的客户红利，上级需要考虑可以提高客户留存率的解决方案
# ##### 无纸化计费的损失率较高，留存客户的支付方式非常normal，说明支付方式并不是留存客户的关键!

# In[ ]:


plot_pie(df)
# 没有家属的用户可能更倾向于流失 => 待验证


# In[482]:


# 直方图
def zf_Draw(df, column):
    plt.figure(figsize=(10,6))
    sns.histplot(data=df, x=column, hue="Churn", multiple="stack", binwidth=0.5)
    plt.title(column + " distribution in Churn")
    plt.show()

# 散点图矩阵
def plot_scatter(df, columns):
    sns.pairplot(df[columns], hue="Churn")
    plt.show()


# In[483]:


zf_Draw(df,'tenure')


# In[484]:


plt.figure(figsize = (20,8))
sns.countplot(x = df.tenure_group , hue = df.Churn,palette=("Blues_d"))
plt.legend(['Non Churn' , 'Churn'])


# # 4.Data modeling

# ### 4.1 数据处理，将非数值形式的数据列变成数值的形式
# #### 4.1.1 二元变量使用lable编码 0 / 1；多元变量，为了避免引入数值大小关系，使用one-hot编码成向量形式

# In[485]:


#二元變數
bin_cols = df.nunique()[df.nunique()==2].keys().tolist()


# In[486]:


bin_cols


# In[487]:


#多元變數
multi_cols = [col for col in cat_cols if col not in bin_cols]
multi_cols


# In[488]:


test_cols = [1,2,3,4,5]
for number in test_cols:
    # do some thing


# In[489]:


# Read in the required kits
# We use label to process the category coding, and logistic must be standardized, and when the data is distributed in EDA, we know that it is not normal, so we continue to use Standard method
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
#二元變數
bin_cols = df.nunique()[df.nunique()==2].keys().tolist()
#多元變數
multi_cols = [col for col in cat_cols if col not in bin_cols]
#將二元數值編碼
# cato = df.tenure_group.cat.codes
# df.tenure_group = cat
le = LabelEncoder()
# df[multi_cols] = df[multi_cols].replace({0:'No' , 1:'Yes'})
# 从 bincols一个个取出值进行操作
for col in bin_cols:
    df[col] = le.fit_transform(df[col])


# In[490]:



df_show = pd.read_csv('/Users/yx/Desktop/code/loss.csv')
multi_cols


# In[491]:


df_show['Contract']


# #### 独热编码，假如一个属性有n种取值，最终编码变成一系列的向量 [ 0 0 ... 1 ... 0 ] 的形式。这个向量长度是n
# #### 不同的位置取1，就表示不同的该变量取值
# #### 如 老虎，狮子，长颈鹿 => [ 1 0 0 ] 老虎 ，[0 1 0] => 狮子
# #### 第一位代表是否是老虎，第二位代表是否是狮子，第三位代表是否是长颈鹿

# In[492]:


test = df_show[['Contract']]


# In[493]:


test


# In[494]:


tt = pd.get_dummies(data = test , columns=['Contract']).astype('int')


# In[495]:


tt
# True 和 False 与 1 / 0等价 不用转换成数字，他们本身就是数字


# In[496]:


df_show[['tenure','MonthlyCharges','TotalCharges']]


# In[497]:


scaled


# In[498]:


# 为了避免引入大小关系，用one-hot encoding
df = pd.get_dummies(data = df , columns=multi_cols)
    
# Handle continuous variables
std = StandardScaler()
scaled = std.fit_transform(df[num_cols])
scaled = pd.DataFrame(scaled,columns=num_cols) 
 
df_origin =  df.copy()
df = df.drop(columns=num_cols , axis = 1)
df = df.merge(scaled , left_index=True , right_index=True , how = 'left')


# In[499]:


df


# #### PCA主成分分析的目的： 使用更少的维度（属性）表示数据，达到降维的目的，也就是找到主要的成分。使用主成分表示原来所有的属性
# #### 也称之为投影（把高维的数据投影，达到降维的目的），投影的目标是 使得降维后的数据之间的方差最大（避免投影后所有的数据黏连到一起，数据失去了区分度
# #### 表明这种投影方式无法完整的表现所有数据的特性）
# #### https://www.zhihu.com/question/41120789/answer/2918798394
# #### 

# In[500]:


from sklearn.decomposition import  PCA
# 整数表示降到的维数
# 小数表示需要保持的信息百分比，维数由算法自己决定 => 使得选取的维数能够保证一定程度的精度
pca = PCA(n_components = 2)
X = df[[col for col in df.columns if col not in Id_col + target_col]]
Y = df[target_col + Id_col]

pc = pca.fit_transform(X)

# 使用逆变换重构数据
X_reconstructed = pca.inverse_transform(pc)
# 计算重构误差
from sklearn.metrics import mean_squared_error
reconstruction_error = mean_squared_error(X, X_reconstructed)

print(reconstruction_error)


# In[501]:


Y


# In[502]:


pca_data


# In[503]:


pca_data  = pd.DataFrame(pc , columns=['PC1' , 'PC2'])
pca_data = pca_data.merge(Y , left_index = True , right_index = True , how = 'left')
pca_data = pca_data.replace({1:'Churn' , 0: 'Not Churn'})


# In[504]:


pca_data


# In[505]:


def pca_scatter(target,color):
    tracer = go.Scatter(x = pca_data[pca_data["Churn"] == target]["PC1"] ,
                        y = pca_data[pca_data["Churn"] == target]["PC2"],
                        name = target,mode = "markers",
                        marker = dict(color = color,
                                      line = dict(width = .5),
                                      symbol =  "diamond-open"),
                        text = ("Customer Id : " + 
                                pca_data[pca_data["Churn"] == target]['customerID'])
                       )
    return tracer
layout = go.Layout(dict(title = "Visualising data with principal components",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                     title = "principal component 1",
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                     title = "principal component 2",
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                        height = 600
                       )
                  )
trace1 = pca_scatter("Churn",'red')
trace2 = pca_scatter("Not Churn",'royalblue')
data = [trace2,trace1]
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)


# In[506]:



bi_cs = bin_cols
dat_rad = df[bin_cols]
#畫出雷達圖
def plot_radar(df,aggregate,title) :
   data_frame = df[df["Churn"] == aggregate]
   
   data_frame_x = data_frame[bi_cs].sum().reset_index()
   data_frame_x.columns  = ["feature","yes"]
   data_frame_x["no"]    = data_frame.shape[0]  - data_frame_x["yes"]
   data_frame_x  = data_frame_x[data_frame_x["feature"] != "Churn"]
   
   #count of 1's(yes)
   trace1 = go.Scatterpolar(r = data_frame_x["yes"].values.tolist(),
                            theta = data_frame_x["feature"].tolist(),
                            fill  = "toself",name = "count of 1's",
                            mode = "markers+lines",
                            marker = dict(size = 5)
                           )
   #count of 0's(No)
   trace2 = go.Scatterpolar(r = data_frame_x["no"].values.tolist(),
                            theta = data_frame_x["feature"].tolist(),
                            fill  = "toself",name = "count of 0's",
                            mode = "markers+lines",
                            marker = dict(size = 5)
                           ) 
   layout = go.Layout(dict(polar = dict(radialaxis = dict(visible = True,
                                                          side = "counterclockwise",
                                                          showline = True,
                                                          linewidth = 2,
                                                          tickwidth = 2,
                                                          gridcolor = "white",
                                                          gridwidth = 2),
                                        angularaxis = dict(tickfont = dict(size = 10),
                                                           layer = "below traces"
                                                          ),
                                        bgcolor  = "rgb(243,243,243)",
                                       ),
                           paper_bgcolor = "rgb(243,243,243)",
                           title = title,height = 700))
   
   data = [trace2,trace1]
   fig = go.Figure(data=data,layout=layout)
   py.iplot(fig)
#plot
plot_radar(dat_rad,1,"Churn -  Customers")
plot_radar(dat_rad,0,"Non Churn - Customers")


# # 二元 => 输出概率
# # 决策阈值 β = 0.5

# In[8]:


from sklearn.model_selection import  train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report
from sklearn.metrics import  roc_auc_score , roc_curve 
from sklearn.metrics import f1_score
import statsmodels.api as sm
from sklearn.metrics import  precision_score ,recall_score
from yellowbrick.classifier import DiscriminationThreshold
# splitting train and test data
# 训练数据 : 预测数据 样本 : 样本外 3:1
train , test = train_test_split(df , test_size = 0.25 , random_state = 3 )

train2 , test2 = train_test_split(df , test_size = 0.25 , random_state = 4 )


cols = [col for col in df.columns if col not in Id_col + target_col]
train_X =train[cols]
train_Y = train[target_col]

test_X = test[cols]
test_Y = test[target_col]


# In[ ]:



def select_model_prediction(algorithm,training_x,testing_x,
                             training_y,testing_y,cols,cf,threshold_plot) :
    #model
    algorithm.fit(training_x,training_y)
    predictions   = algorithm.predict(testing_x)

 
    probabilities = algorithm.predict_proba(testing_x)
    #coeffs
    if   cf == "coefficients" :
        coefficients  = pd.DataFrame(algorithm.coef_.ravel())
    elif cf == "features" :
        coefficients  = pd.DataFrame(algorithm.feature_importances_)

    column_df     = pd.DataFrame(cols)
    coef_sumry    = (pd.merge(coefficients,column_df,left_index= True,
                              right_index= True, how = "left"))
    coef_sumry.columns = ["coefficients","features"]
    coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)

    print(algorithm)
    print("\n Classification report : \n",classification_report(testing_y,predictions))
    print("Accuracy   Score : ",accuracy_score(testing_y,predictions))
    #confusion matrix
    conf_matrix = confusion_matrix(testing_y,predictions)
    #roc_auc_score
    model_roc_auc = roc_auc_score(testing_y,predictions) 
    print("Area under curve : ",model_roc_auc,"\n")
    fpr,tpr,thresholds = roc_curve(testing_y,probabilities[:,1])

    #plot confusion matrix
    trace1 = go.Heatmap(z = conf_matrix ,
                        x = ["Not churn","Churn"],
                        y = ["Not churn","Churn"],
                        showscale  = False,colorscale = "Picnic",
                        name = "matrix")

    #plot roc curve
    trace2 = go.Scatter(x = fpr,y = tpr,
                        name = "Roc : " + str(model_roc_auc),
                        line = dict(color = ('rgb(22, 96, 167)'),width = 2))
    trace3 = go.Scatter(x = [0,1],y=[0,1],
                        line = dict(color = ('rgb(205, 12, 24)'),width = 2,
                        dash = 'dot'))

    #plot coeffs
    trace4 = go.Bar(x = coef_sumry["features"],y = coef_sumry["coefficients"],
                    name = "coefficients",
                    marker = dict(color = coef_sumry["coefficients"],
                                  colorscale = "Picnic",
                                  line = dict(width = .6,color = "black")))

    #subplots
    fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                            subplot_titles=('Confusion Matrix',
                                            'Receiver operating characteristic',
                                            'Feature Importances'))

    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,2)
    fig.append_trace(trace3,1,2)
    fig.append_trace(trace4,2,1)

    fig['layout'].update(showlegend=False, title="Model performance" ,
                         autosize = False,height = 900,width = 800,
                         plot_bgcolor = 'rgba(240,240,240, 0.95)',
                         paper_bgcolor = 'rgba(240,240,240, 0.95)',
                         margin = dict(b = 195))
    fig["layout"]["xaxis2"].update(dict(title = "false positive rate"))
    fig["layout"]["yaxis2"].update(dict(title = "true positive rate"))
    fig["layout"]["xaxis3"].update(dict(showgrid = True,tickfont = dict(size = 10),
                                        tickangle = 90))
    py.iplot(fig)

    #用yellow_brick幫我們可視化圖片
    if threshold_plot == True : 
        visualizer = DiscriminationThreshold(algorithm)
        visualizer.fit(training_x,training_y)
        visualizer.poof()


# In[ ]:


# 决策阈值 => 0.5
# 决策阈值是否应该是0.5

# 决策阈值设置的值 影响 模型的分类结果
#               [ 0    0    0     0     1    1 ]
# prediction = [ 0.3, 0.4, 0.4, 0.51, 0.8, 0.9]

# => 0.5 => RESULT = 0 , 0 , 0 ,  1,    1  ,  1 ] 5/6
# => 0.6           = 0   0   0    0     1     1   6/6




# In[508]:


#寫好logistic的演算法，正則我們採用Ridge算法
logit  = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
  intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
  penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
  verbose=0, warm_start=False)
#跑model
select_model_prediction (logit,train_X,test_X,train_Y,test_Y,
                     cols,"coefficients",threshold_plot = True)


# In[411]:


# 使用模型进行预测
predicted_values = logit.predict(test_X)

# 打印预测值
print(predicted_values)


# In[509]:



class_weight1 = {0:1 , 1:2.5}
logit  = LogisticRegression(C=1.0, class_weight=class_weight1, dual=False, fit_intercept=True,
 intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
 penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
 verbose=0, warm_start=False)
#跑model
select_model_prediction (logit,train_X,test_X,train_Y,test_Y,
                    cols,"coefficients",threshold_plot = True)

