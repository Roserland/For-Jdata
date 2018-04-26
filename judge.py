
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#用户评价函数
#预测集传入为DataFrame, pred_DataFrame.columns = ["user_id", "pred_date"]
#验证集传入为DataFrame, comfirm_data.columns = ["user_id", "Min_date"]
def User_Judgement(pred_DataFrame, comfirm_data):
    #The Wi part
    length = len(pred_DataFrame)
    user_index = np.arange(length)
    offset = np.ones(length)
    Wi = 1 / (offset + np.log(user_index + offset))
    Wi_sum = Wi.sum()
    #The Oi part
    merge_data = pd.merge(pred_DataFrame, comfirm_data, how = 'inner', on = 'user_id')   #user_id 的顺序问题？
    merge_pred_date = np.array(merge_data["pred_date"]).astype(int)
    merge_Min_date = np.array(merge_data['Min_date']).astype(int)
    Oi_array = (merge_pred_date == merge_Min_date)
    Oi_int = Oi_array.astype(int)
    result_array = Oi_int * Wi
    S1 = result_array.sum()  / Wi_sum
    return S1


# In[3]:


#用户评价函数
#预测集传入为DataFrame, pred_DataFrame.columns = ["user_id", "pred_date"]
#验证集传入为DataFrame, comfirm_data.columns = ["user_id", "Min_date"]
def User_Order_Date_Judgement(pred_DataFrame, comfirm_data):
    merge_data = pd.merge(pred_DataFrame, comfirm_data, how = 'inner', on = "user_id")
    hit_num = len(merge_data)
    merge_pred_date = np.array(merge_data["pred_date"]).astype(int)
    merge_Min_date = np.array(merge_data["Min_date"]).astype(int)
    diff_array = merge_Min_date - merge_pred_date
    du2_array = diff * diff
    fu_array = (np.ones(hit_num)*10)/ (np.ones(hit_num)*10 + du2_array)
    fu_sum = fu_array.sum()
    S2 = fu_sum / hit_num
    return S2


# In[4]:


#预测CSV转化函数，转化后为DataFrame,  columns = ["user_id", "pred_date"]
def csv_transfer(csv_filename): 
    file = pd.read_csv(csv_filename)
    pred_date_list = file["pred_date"].tolist()
    date_list = []
    for x in pred_date_list:
        a = x.split('-')[2]
        date_list.append(a)
    length = len(date_list)
    user_id = np.ones(length) + np.array(range(length))
    user_id = user_id.astype(int)
    Data = {"user_id":user_id,"pred_date":np.array(date_list)}   #索引可能有问题
    Data = pd.DataFrame(Data)
    return Data

