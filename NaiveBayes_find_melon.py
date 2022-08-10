#朴素贝叶斯

import pandas as pd
import numpy as np
import random


class NaiveBayes():
    def __init__(self,data,X,y):
        self.data=data  #数据,为pandas类型
        self.X=X        #特征矩阵
        self.y=y        #标签
        self.N=len(y)   #样本数量


    def NB_fit(self,X,y):
        #使用unique提取分类的种类，此处为好瓜（1）和坏瓜（0）
        #unique函数，返回对象中不重复的值
        classess=y[y.columns[0]].unique()
        
        #使用counts函数求各个类别出现个数
        #value_counts是pandas统计某列各值个数
        class_count=y[y.columns[0]].value_counts()
        #计算类先验（cpp）
        class_prior=class_count/self.N
        
        #计算各特征先验，预测先验概率（PPP）
        x_prior=dict()
        for feature in range(X.shape[1]):
            #x_class_prior为各特征的先验概率
            x_class_prior=X[X.columns[feature]].value_counts()/self.N
            x_prior[x_class_prior.name]=x_class_prior


        #计算似然
        #p_x_y表示（各个特征，标签）出现的概率
        prior=dict()
        for feature in X.columns:
            for label in classess:
                #X[(y==j).values][feature]即X中列名为feature的列中对应的y=j的所有行，是一个Series类型
                #.value_counts()计算各（特征值，标签）出现的次数
                p_x_y=X[(y==label).values][feature].value_counts()
                #p_x_y是一个Series类型
                
                #.index为p_x_y的索引值或第一列，此处的第一列为特征值
                for i in p_x_y.index:
                    #(feature,i,label)是字典元素的键
                    #p_x_y[i]指索引为i的值，这里指特征值为i出现的个数
                    #class_count[label]指索引为label的值，这里指标签为label出现的个数
                    prior[(feature,i,label)]=p_x_y[i]/class_count[label]
                    
        #       标签，类先验（cpp)，似然，特征先验（ppp）
        return classess,class_prior,prior,x_prior
      


    #先验概率计算
    def train(self,X,y):
        return self.NB_fit(X,y)


    #per_predict是预测单个样本的函数
    #X_test为测试集，是各特征名：特征值为元素组成的字典
    def per_predict(self,X_test,classess,class_prior,prior,x_prior):
        res=[]
        #进行预测
        for c in classess:
            #获取标签c的类先验概率
            p_y=class_prior[c]
            #初始化（特征值，标签）的概率为1
            p_x_y=1

            #测试集检测
            #.items()为对象中各元素，此处指该样本的所有特征
            for feature in X_test.items():
                #feature为测试集中的特征矩阵
                #(feature,标签)组成的元组可以作为似然prior的索引
                index=tuple(list(feature)+[c])  
                
                #x_prior是{特征名：（该特征值，标签）出现的概率}组成的字典
                #x_prior[index[0]][index[1]]，index[0]指特征名，index[1]指该测试集样本的单个特征值，返回的是其对应的特征先验概率
                #训练集中可能不含该组合，其似然应为0
                if index in prior.keys():
                    p_x_y*=(prior[index]*p_y)/x_prior[index[0]][index[1]]
                else:
                    p_x_y*=0

            #res中存储的是该样本是好瓜和坏瓜的概率
            res.append(p_x_y)
        return classess[np.argmax(res)]



    #测试集生成
    def wash(self,data):
        #提取特征名
        header=data.columns

        #将data转化成矩阵，使用shuffle打乱顺序
        base=np.array(data)
        np.random.shuffle(base)

        #提取测试集特征矩阵
        X_test=base[:,1:7]
        y_test=base[:,-1]
        
        return header,X_test,y_test
        

    #全测试集测试
    def predict(self,data,X,y):
        #生成测试集
        header,simple,y_test=self.wash(data)
        
        #获得先验概率
        classess,class_prior,prior,x_prior=self.train(X,y)
        
        #进行预测
        pred=[]
        for sim in np.array(simple):
            #X_test为各个样本的{特征名：特征值}组成的字典
            X_test=dict()
            for i in range(len(sim)):
                X_test[header[i+1]]=sim[i]
            #预测单个样本的标签
            pred.append(self.per_predict(X_test,classess,class_prior,prior,x_prior))

        #计算正确率
        count=sum(pred==y_test)
                  
        return count
         
   


#读取数据
data=pd.DataFrame(pd.read_excel('melon.xlsx'))


#简化数据
data=data.replace(['浅白','青绿','乌黑',\
              '硬挺','稍蜷','蜷缩',\
              '清脆','浊响','沉闷',\
              '清晰','稍糊','模糊',\
              '平坦','稍凹','凹陷',\
              '硬滑','软粘','是','否'],[0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,1,0])


def main():
    X=data.iloc[:,1:7]
    y=data.iloc[:,-1:]  #使用切片时会返回DataFrame类型
    
    #创建筛选器Huaqiang
    Huaqiang=NaiveBayes(data,X,y)

    #进行训练，测试
    pred=Huaqiang.predict(Huaqiang.data,Huaqiang.X,Huaqiang.y)
    


main()

