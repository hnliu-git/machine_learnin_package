#coding=utf-8
'''
@author: Teren
机器学习方法预测以及基础调参
'''

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier


output = 'baseline_SVC_goodtfidf_charword_nmf.csv'
index = True
index_num = 100
model_name = 'LR'

"""
数据输入
"""
data = open('../data/data_nmf_good_charword_30w.pkl','rb')
import pickle
x_train,y_train ,x_test = pickle.load(data)



class SKModel:

    def __init__(self,x_train,y_train,x_test,model,output):
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.output=output
        clf = self.train_model(model)
        self.y_predict = clf.predict(self.x_test)

    def train_model(self,model):
        if model=='NB':
            clf = MultinomialNB(alpha=1)
            params = {'alpha': [0.001]}
            grid = GridSearchCV(clf, params, scoring='f1_micro')
            grid.fit(self.x_train, self.y_train)
            print (grid.best_score_)
            print (grid.best_params_)
            clf = grid.best_estimator_
        elif model == 'LR':
            clf = LogisticRegression() #0.75  #C>=100 select #C>=100 endsemble #C>=100 cutcharwordselect
            params = {'C': [1, 10, 100]}
            grid = GridSearchCV(clf, params, scoring='f1_micro')
            grid.fit(self.x_train, self.y_train)
            print (grid.best_score_)
            print (grid.best_params_)
            clf = grid.best_estimator_
        elif model == 'SVM':
            clf = SVC()
            clf.fit(self.x_train,self.y_train)
        elif model == 'Softmax':
            clf = LogisticRegression(multi_class='multinomial',solver='lbfgs')  # 0.77 3rd C=10
            params = {'C': [1,10,100]}
            grid = GridSearchCV(clf, params, scoring='f1_micro')
            grid.fit(self.x_train, self.y_train)
            print (grid.best_score_)
            print (grid.best_params_)
            clf = grid.best_estimator_
        elif model == 'LinearSVC': #C=1 0.77 1st tfidf #C=100 lsa
            clf = LinearSVC(C=1,max_iter=1000,verbose=1,multi_class='ovr')
            params = {'C':[1,10]}
            grid = GridSearchCV(clf,params,scoring='f1_micro',n_jobs=-1)
            grid.fit(self.x_train,self.y_train)
            print (grid.best_score_)
            print (grid.best_params_)
            clf = grid.best_estimator_
        elif model =='AdaLR':
            bclf = LogisticRegression(C=100)
            clf = AdaBoostClassifier(base_estimator=bclf,learning_rate=1)
            params = {'learning_rate': [0.1]}
            grid = GridSearchCV(clf, params, scoring='f1_micro', n_jobs=-1)
            grid.fit(self.x_train, self.y_train)
            print (grid.best_score_)
            print (grid.best_params_)
            clf = grid.best_estimator_
        return clf


    def write_predict(self):
        with open(self.output, 'a') as wf:
            wf.write("id,class\n")
            cot = 0
            for i in self.y_predict:
                wf.write(str(cot) + "," + str(i+1) + "\n")
                cot += 1

if index:
    SKModel(x_train[:index_num],y_train[:index_num],x_test[:index_num],model_name,output).write_predict()
else:
    SKModel(x_train, y_train, x_test, model_name, output).write_predict()
print (output)







