#coding=utf8
"""
@author: Teren
基于CSV做投票ensemble
另:
可基于sklearn.ensemble import VotingClassifier做投票ensemble
"""

import pandas as pd


class_weight = {1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1,10:1,11:1,12:1,13:1,14:1,15:1,16:1,17:1,18:1,19:1,20:1}
model_weight = [1,1,1,1,1]
model_list = ['BASELINE.csv','baseline_SVC_select_NMF.csv','data_ensemble.pkl_lgb.csv']
length = 102276
output_path = 'vote.csv'

'''
读取文件
'''
def get_df(csv_l):
    df_list = []
    for csv in csv_l:
        df_list.append(pd.read_csv('../baseline/'+csv))
    return df_list

'''
通过评分排序获取每个ID对应的class
'''
def get_class(vote_dict):
    items = vote_dict.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort(reverse=True)
    return backitems[0][1]

'''
投票
'''
def vote(df_list,weight_list,rows_num,classweight):
    y_pred = []
    for i in range(0,rows_num+1):
        vote_dict ={}
        for j,df in enumerate(df_list):
            if df['class'][i] in vote_dict:
                vote_dict[df['class'][i]] = float(weight_list[j])*classweight[int(df['class'][i])]+vote_dict[df['class'][i]]
            else:
                vote_dict[df['class'][i]] = float(weight_list[j])*classweight[int(df['class'][i])]
        pred = get_class(vote_dict)
        y_pred.append(int(pred)-1)
        print (i,pred)
    return y_pred

'''
输出
'''
def write_predict(filepath,y_pred):
    with open(filepath, 'a') as wf:
        wf.write("id,class\n")
        cot = 0
        for i in y_pred:
            wf.write(str(cot) + "," + str(i+1) + "\n")
            cot += 1

write_predict(output_path,vote(get_df(model_list),model_weight,length,class_weight))