#!/usr/bin/env python
# encoding: utf-8
# @File  : my_svm1.py
# @Author: zhangyangyang
# @Date  : 2019/4/18 19:20
# @Software: PyCharm
# @desc:
import jieba
import os
import pickle  # 持久化
from numpy import *
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer  # TF_IDF向量生成类
from sklearn.datasets.base import Bunch
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn import metrics

import numpy


# 读文件
def read_file(file_path):
    with open(file_path, 'r', errors='ignore', encoding='utf-8') as file:
        content = file.read()
        return content


# 写文件
def save_file(file_path, result):
    with open(file_path, 'w', errors='ignore', encoding='utf-8') as file:
        file.write(result)


# 分词
def seg_text(input_path, result_path, stop_word):
    father_lists = os.listdir(input_path)  # 主目录
    for eachDir in father_lists:  # 遍历主目录中各个文件夹
        each_path = input_path + eachDir + "/"  # 保存主目录中每个文件夹目录，便于遍历二级文件
        each_result_path = result_path + eachDir + "/"  # 分词结果文件存入的目录
        if not os.path.exists(each_result_path):
            os.makedirs(each_result_path)
        child_lists = os.listdir(each_path)  # 获取每个文件夹中的各个文件
        for each_file in child_lists:  # 遍历每个文件夹中的子文件
            each_path_file = each_path + each_file  # 获得每个文件路径
            #  print(eachFile)
            content = read_file(each_path_file)  # 调用上面函数读取内容
            # content = str(content)
            result = (str(content)).replace("\r\n", "").strip()  # 删除多余空行与空格
            # result = content.replace("\r\n","").strip()

            cut_result = jieba.cut(result)  # 默认方式分词，分词结果用空格隔开
            out_word = ''
            for word in cut_result:
                if word not in stop_word:
                    if word != '\t':
                        out_word += word
                        out_word += ' '
            save_file(each_result_path + each_file, out_word)  # 调用上面函数保存文件


def bunch_save(input_file, output_file):
    cate_list = os.listdir(input_file)
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(cate_list)  # 将类别保存到Bunch对象中
    for eachDir in cate_list:
        each_path = input_file + eachDir + "/"
        file_list = os.listdir(each_path)
        for eachFile in file_list:  # 二级目录中的每个子文件
            full_name = each_path + eachFile  # 二级目录子文件全路径
            bunch.label.append(eachDir)  # 当前分类标签
            bunch.filenames.append(full_name)  # 保存当前文件的路径
            bunch.contents.append(read_file(full_name).strip())  # 保存文件词向量
    with open(output_file, 'wb') as file_obj:  # 持久化必须用二进制访问模式打开
        pickle.dump(bunch, file_obj)
        # pickle.dump(obj, file, [,protocol])函数的功能：将obj对象序列化存入已经打开的file中。
        # obj：想要序列化的obj对象。
        # file:文件名称。
        # protocol：序列化使用的协议。如果该项省略，则默认为0。如果为负值或HIGHEST_PROTOCOL，则使用最高的协议版本


def read_bunch(file_path):
    with open(file_path, 'rb') as file:
        bunch = pickle.load(file)
        # pickle.load(file)
        # 函数的功能：将file中的对象序列化读出。
    return bunch


def write_bunch(file_path, bunch_file):
    with open(file_path, 'wb') as file:
        pickle.dump(bunch_file, file)


# 获得停用词
def get_stop_word(input_file):
    stop_word = read_file(input_file).splitlines()
    return stop_word


# 求得TF-IDF向量
def get_tfidf_mat(input_path, output_path):
    bunch = read_bunch(input_path)
    tfidf_space = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                        vocabulary={})
    # 初始化向量空间
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    # transformer = TfidfTransformer()  # 该类会统计每个词语的TF-IDF权值
    # 文本转化为词频矩阵，单独保存字典文件
    tfidf_space.tdm = tfidf_vectorizer.fit_transform(bunch.contents)
    tfidf_space.vocabulary = tfidf_vectorizer.vocabulary_  # 获取词汇
    write_bunch(output_path, tfidf_space)
    # 输出特征矩阵
    # test_data = tfidf_space.tdm[1]
    # print(size(test_data))
    # print(test_data)


def get_test_space(test_set_path, train_space_path, test_space_path):
    bunch = read_bunch(test_set_path)
    # 构建测试集TF-IDF向量空间
    test_space = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})
    # 导入训练集的词袋
    train_bunch = read_bunch(train_space_path)
    # 使用TfidfVectorizer初始化向量空间模型  使用训练集词袋向量
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                       vocabulary=train_bunch.vocabulary)
    # transformer = TfidfTransformer()
    test_space.tdm = tfidf_vectorizer.fit_transform(bunch.contents)
    test_space.vocabulary = train_bunch.vocabulary
    # 持久化
    write_bunch(test_space_path, test_space)


# svm训练
# ‘linear’:线性核函数
# ‘poly’：多项式核函数
# ‘rbf’：径像核函数/高斯核
# ‘sigmod’:sigmod核函数
# ‘precomputed’:核矩阵
def svm_algorithm(train_path, test_path):
    clf = SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
              gamma=1, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
    train_set = read_bunch(train_path)
    test_set = read_bunch(test_path)
    clf.fit(train_set.tdm, train_set.label)
    # 调用自动寻参，获得最佳模型
    # clf = svm_cross_validation(train_set.tdm, train_set.label)
    print(clf.score(test_set.tdm, test_set.label))
    # 保存模型文件
    joblib.dump(clf, "svm_model.m")
    re = clf.predict(test_set.tdm)
    print(len(test_set.label))
    print(len(re))
    evaluate(numpy.asarray(test_set.label), re)
    target_names = ['动画', '动作', '剧情', '喜剧']
    xx = metrics.classification_report(test_set.label, re, target_names=target_names)
    print(xx)


# 得到准确率和召回率
def evaluate(actual, pred):
    m_precision = metrics.precision_score(actual, pred, average='macro')
    m_recall = metrics.recall_score(actual, pred, average='macro')
    m_f1_score = metrics.f1_score(actual, pred, average='weighted')
    m_fbeta_score = metrics.fbeta_score(actual, pred, beta=0.5, average='macro')
    print('precision:{0:.3f}'.format(m_precision))
    print('recall:{0:0.3f}'.format(m_recall))
    print('f1_score:{0:.3f}'.format(m_f1_score))
    print('fbeta_score:{0:.3f}'.format(m_fbeta_score))


#  网格搜索-参数寻优
def svm_cross_validation(train_x, train_y):
    model = SVC(kernel='rbf', probability=True)
    thresholds = numpy.linspace(0, 0.001, 100)  # 设置gamma参数列表
    param_grid = {'gamma': thresholds}
    grid_search = GridSearchCV(model, param_grid, cv=2)
    grid_search.fit(train_x, train_y)
    print("最佳效果：%0.3f" % grid_search.best_score_)
    print("最优参数组合：")

    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print('\t%s:%r' % (param_name, best_parameters[param_name]))
    return model


if __name__ == "__main__":
    # 分词，第一个是分词输入，第二个参数是结果保存的路径
    stop_words = get_stop_word("E:/python-data1/stop/stopword.txt")
    seg_text("E:/python-data1/data/", "E:/python-data1/segResult/", stop_words)
    # 输入分词，输出分词向量
    bunch_save("E:/python-data1/segResult/", "E:/python-data1/train_set.dat")
    # 输入词向量，输出特征空间
    get_tfidf_mat("E:/python-data1/train_set.dat", "E:/python-data1/tfidfspace.dat")

    # 测试集-分词
    seg_text("E:/python  -data1/test/", "E:/python-data1/test_segResult/", stop_words)
    bunch_save("E:/python-data1/test_segResult/", "E:/python-data1/test_set.dat")
    get_test_space("E:/python-data1/test_set.dat", "E:/python-data1/tfidfspace.dat",
                   "E:/python-data1/testspace.dat")
    svm_algorithm("E:/python-data1/tfidfspace.dat", "E:/python-data1/testspace.dat")
