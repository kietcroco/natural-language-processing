# libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pandas
from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import heapq
import logisticregression as lgr
import copy
from pprint import pprint

np.random.seed(0)
dfs = []
dfs.append(pandas.read_csv('ThoiSu.csv'))
dfs.append(pandas.read_csv('TheThao.csv'))
dfs.append(pandas.read_csv('TheGioi.csv'))
df = pandas.concat(dfs, ignore_index=True)

###### 1. Thu thập một số bài báo từ các thể loại khác nhau
# đếm số dòng ThoiSu
size_1 = df[ df['label'] == 'ThoiSu' ].shape[0]
# đếm số dòng TheThao
size_2 = df[ df['label'] == 'TheThao' ].shape[0]
# đếm số dòng TheGioi
size_3 = df[ df['label'] == 'TheGioi' ].shape[0]

# print('Số tài liệu ThoiSu: %s' %size_1)
# print('Số tài liệu TheThao: %s' %size_2)
# print('Số tài liệu TheGioi: %s' %size_3)

# tách dữ liệu thành 2 tập training và test: train va test
train_x, test_x, train_y, test_y = model_selection.train_test_split(df['text'], df['label'])

# đưa label ve 0 va 1
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

# print('chuyển label ["ThoiSu", "TheThao", "TheGioi"] thành %s' %encoder.transform(["ThoiSu", "TheThao", "TheGioi"]))

###### 2. Thu thập một số bài báo từ các thể loại khác nhau
# word level tf-idf
tfidf_vect = TfidfVectorizer(encoding='utf-16', analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(df['text'])

xtrain_tfidf =  tfidf_vect.transform(train_x)
xtest_tfidf =  tfidf_vect.transform(test_x)

# print('Số tài liệu training: %s' %str(xtrain_tfidf.shape[0]))
# print('Số tài liệu testing: %s' %str(xtest_tfidf.shape[0]))
# print('Number of features of each document: %s' %str(xtrain_tfidf.shape[1]))

# Many software bugs in deep learning come from having matrix/vector dimensions that don't fit. If you can keep your matrix/vector dimensions straight you will go a long way toward eliminating many bugs.
# Exercise:: For convenience, you should now transpose the training and testing np-array, and expand the shape of the lable arrays in the axis=0 position.
# After this, our training (and test) dataset is a np-array where each column represents a document vector. There should be the number of training documents (respectively the number of testing documents) as the number of columns.

train_y = np.expand_dims(train_y, axis=0)
test_y = np.expand_dims(test_y, axis=0)
# print('===============')
# convert sparse to dense matrix 
xtrain_tfidf =  xtrain_tfidf.T.toarray() 
xtest_tfidf =  xtest_tfidf.T.toarray()

# dict = tfidf_vect.vocabulary_ # Shape {word : index}
# pprint(xtrain_tfidf.shape[1])

# for i in range(xtrain_tfidf.shape[1]):

#     label = encoder.inverse_transform([train_y[0, i]])
#     priorityQueue = []

#     for word in dict:

#         score = xtrain_tfidf[dict[word], i]
#         heapq.heappush(priorityQueue, (-score, word))

#     top10 = heapq.nsmallest(10, priorityQueue)
#     wordList = [top10[i][1] for i in range(10)]
#     print(i, wordList, label)

def classify(xtrain_tfidf_, dataset_train_y_, xtest_tfidf_, dataset_test_y_, number_):
    train_y = copy.deepcopy(dataset_train_y_)
    test_y = copy.deepcopy(dataset_test_y_)
    for i in range(len(train_y[0])):
        if train_y[0, i] == number_:
            train_y[0, i] = 0
        else:
            train_y[0, i] = 1
    for i in range(len(test_y[0])):
        if test_y[0, i] == number_:
            test_y[0, i] = 0
        else:
            test_y[0, i] = 1
    d = lgr.model(xtrain_tfidf_, train_y, xtest_tfidf_, test_y, num_iterations = 3000, learning_rate = .75, print_cost = False)
    return d

d0 = classify(xtrain_tfidf, train_y, xtest_tfidf, test_y, 0)
d1 = classify(xtrain_tfidf, train_y, xtest_tfidf, test_y, 1)
d2 = classify(xtrain_tfidf, train_y, xtest_tfidf, test_y, 2)

Y_train_prediction = np.zeros((1, xtrain_tfidf.shape[1]), dtype=np.int)
Y_test_prediction = np.zeros((1, xtest_tfidf.shape[1]), dtype=np.int)

pprint(xtrain_tfidf.shape[1])

# for i in range(Y_train_prediction.shape[1]):
#     Y_train_prediction[0, i] = np.argmin(
#         np.array(
#             [
#                 d0['sigmoid_result_train'][0, i],
#                 d1['sigmoid_result_train'][0, i],
#                 d2['sigmoid_result_train'][0, i]
#             ]
#         )
#     )
# for i in range(Y_test_prediction.shape[1]):
#     Y_test_prediction[0, i] = np.argmin(
#         np.array(
#             [
#                 d0['sigmoid_result_test'][0, i],
#                 d1['sigmoid_result_test'][0, i],
#                 d2['sigmoid_result_test'][0, i]
#             ]
#         )
#     )

# print("Final Train Accuracy: {} %".format(100 - np.mean(np.abs(Y_train_prediction - train_y)) * 100))
# print("Final Test Accuracy: {} %".format(100 - np.mean(np.abs(Y_test_prediction - test_y)) * 100))

# d = lgr.model(xtrain_tfidf, train_y, xtest_tfidf, test_y, num_iterations = 3000, learning_rate = .5, print_cost = True)