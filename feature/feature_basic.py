# -*- coding: utf-8 -*-
"""
@brief : 将原始数据数字化为各类特征，并将结果保存至本地
@author: Teren
"""
import pandas as pd
import pickle
import time

t_start = time.time()

feature_name = "Tfidf"
feature_num = 10000
Index = False
index_num = 2000
drop_column = 'article'
data_column = 'word_seg'
y_column = 'class'
output = "here.pkl"
"""
输入数据,得到df_all与df_test
"""
df_train = pd.read_csv('../data/train_set.csv')
df_train.drop(columns=drop_column, inplace=True)
df_test = pd.read_csv('../data/test_set.csv')
df_test.drop(columns=drop_column, inplace=True)
# Important!!!!!!!!!!!!!
df_all = pd.concat(objs=[df_train, df_test], axis=0, sort=True)
# Important!!!!!!!!!!!!!
y_train = (df_train[y_column] - 1).values

'''
前后2000的数据
'''
if Index:
    for i,line in enumerate(df_all[data_column]):
        group = line.split(' ')
        if len(group)>(index_num*2):
            df_all[data_column][i] = ' '.join(group[:index_num]+group[-index_num:])


"""
选择构造特征的方法
"""

def CreateFeature(feature_name,feature_num,df_train,df_test):
    if feature_name == "Tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True, max_features=feature_num)
        vectorizer.fit(df_all[data_column])
        x_train = vectorizer.transform(df_train[data_column])
        x_test = vectorizer.transform(df_test[data_column])
    elif feature_name == "Tf":
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=100, max_df=0.8)
        vectorizer.fit(df_all[data_column])
        x_train = vectorizer.transform(df_train[data_column])
        x_test = vectorizer.transform(df_test[data_column])
    elif feature_name == "Hash":
        from sklearn.feature_extraction.text import HashingVectorizer
        vectorizer = HashingVectorizer(ngram_range=(1, 2), n_features=200)
        df_train = vectorizer.fit_transform(df_train[data_column])
        x_train = df_train[:len(y_train)]
        x_test = df_train[len(y_train):]
    return x_train,x_test

"""=====================================================================================================================
    3 保存至本地
"""
def SaveFeature(x_train,y_train,x_test,output):
    data = (x_train, y_train, x_test)
    fp = open(output, 'wb')
    pickle.dump(data, fp)
    fp.close()
    print("保存到本地%s"%output)


t_end = time.time()
print("已将原始数据数字化为%s特征，共耗时：{}min".format((t_end - t_start) / 60)%feature_name)

x_train,x_test = CreateFeature(feature_name,feature_num,df_all,df_test)
SaveFeature(x_train,df_train,x_test,output)