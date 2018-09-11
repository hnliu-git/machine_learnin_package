# -*- coding: utf-8 -*-
"""
@brief : 特征降维，并将结果保存至本地
@author: Teren
"""
import pickle
import time

t_start = time.time()

feature_name = "nmf"
feature_num = 10000
Index = False
index_num = 2000
output = "here.pkl"

"""
输入数据,得到x_train,x_test,y_train
"""
data_path = './data_tfidf_select_LSVC_l2_901288.pkl'
data = open(data_path, 'rb')
x_train, y_train, x_test = pickle.load(data)
data.close()

"""
选择构造特征的方法
"""

def CreateFeature(feature_name,feature_num,x_train,x_test):
    if feature_name == "lda":
        from sklearn.decomposition import LatentDirichletAllocation
        lda = LatentDirichletAllocation(n_components=feature_num)
        x_train = lda.fit_transform(x_train)
        x_test = lda.transform(x_test)
    elif feature_name == "lsa":
        from sklearn.decomposition import TruncatedSVD
        lsa = TruncatedSVD(n_components=feature_num)
        x_train = lsa.fit_transform(x_train)
        x_test = lsa.transform(x_test)
    elif feature_name == "nmf":
        from sklearn.decomposition import NMF
        nmf = NMF(n_components=feature_num)
        x_train = nmf.fit_transform(x_train)
        x_test = nmf.transform(x_test)


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
x_train,x_test = CreateFeature(feature_name,feature_num,x_train,x_test)
SaveFeature(x_train,y_train,x_test,output)