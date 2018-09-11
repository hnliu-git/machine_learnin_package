#coding=utf8

"""
@author: Teren
普通Stacking
"""

import numpy as np
import blue_prepocessing
from sklearn.model_selection import KFold
from  sklearn.ensemble import  RandomForestClassifier

index = True
index_num = 100
base_models = [('LR',1),('Softmax',10),('SVM',1),('SVM',100)]
stacker = RandomForestClassifier()
k_fold = 5
output = 'stack_boost.csv'

"""
数据读入
"""
data_fp_tfidf_select = open('../data/data_tfidf_select_LSVC_l2_901288.pkl', 'rb')
import pickle
x_train, y_train, x_test = pickle.load(data_fp_tfidf_select)



"""
选择基分类器
@:param 分类器的参数
"""
def SelectModel(modelname,param):

    if modelname == "SVM":
        from sklearn.svm import LinearSVC
        model = LinearSVC(C=param)
    elif modelname == "GBDT":
        from  sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier()
    elif modelname == "RF":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
    elif modelname == "KNN":
        from sklearn.neighbors import KNeighborsClassifier as knn
        model = knn()
    elif modelname == "LR":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=param)
    elif modelname == 'NB':
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB(alpha=1)
    elif modelname == "Softmax":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(multi_class='multinomial',solver='lbfgs',C=param)
    return model

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


"""
融合类
"""
class Ensemble(object):

    def __init__(self,n_folds,stacker,base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self,x,y,t):


        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=2016).split(x)
        # for j,(train_idx,test_idx) in enumerate(folds):
        #     print j,'train',train_idx,'test',test_idx

        s_train = np.zeros((x.shape[0],len(self.base_models)))
        s_test = np.zeros((t.shape[0],len(self.base_models)))


        for i,clf in enumerate(self.base_models):
            print ("Stack%d...."%i)
            s_test_i = np.zeros((t.shape[0],self.n_folds))

            for j,(train_idx,test_idx) in enumerate(folds):
                x_train = x[train_idx]
                y_train = y[train_idx]
                x_holdout = x[test_idx]
                clf.fit(x_train,y_train)
                y_pred = clf.predict(x_holdout)[:]
                s_train[test_idx,i] = y_pred
                s_test_i[:,j] = clf.predict(t)[:]

            s_test[:,i] = s_test_i.mean(1)

        print ("Stacker Fitting")

        clf = self.stacker.fit(s_train,y)
        y_pred = clf.predict(s_test)[:]

        return y_pred

if __name__ == "__main__":
    stacking = Ensemble(k_fold,stacker,[SelectModel(i,j) for i,j in base_models])
    if index:
        write_predict(str(index_num)+output,stacking.fit_predict(x_train[:index_num],y_train[:index_num],x_test[:index_num]))
    else:
        write_predict(output, stacking.fit_predict(x_train, y_train, x_test))