import csv
import sys
import numpy
from scipy.sparse import csr_matrix
from sklearn.svm.classes import SVC
import Features_manager
import Database_manager
import joblib
from sklearn.metrics.classification import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold



def unique(list1):
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
            # print list
    for x in unique_list:
        print(x)


# initialize database_manager
database_manager = Database_manager.make_database_manager("en")
# initialize feature_manager
feature_manager = Features_manager.make_feature_manager()

# recover tweets
tweets = numpy.array(database_manager.return_tweets_training())
labels = numpy.array(feature_manager.get_label(tweets))

unique(labels)

print(len(tweets))


# recover keyword list corresponding to available features
feature_types = feature_manager.get_availablefeaturetypes()
"""
or you could include only desired features
feature_type=[
            "ngrams",
            "ngramshashtag",
            "chargrams",
            "numhashtag",
            "puntuactionmarks",
            "length",
            ]
"""
# create the feature space with all available features
X, feature_names, feature_type_indexes = feature_manager.create_feature_space(tweets, feature_types)


print("features:", feature_types)
print("feature space dimension:", X.shape)
print(())

golden = []
predict = []
kf = KFold(n_splits=5, shuffle=True, random_state=True)

for index_train, index_test in kf.split(X):

    clf = SVC(kernel="linear")

    clf.fit(X[index_train], labels[index_train])
    test_predict = clf.predict(X[index_test])

    golden = numpy.concatenate((golden, labels[index_test]), axis=0)
    predict = numpy.concatenate((predict, test_predict), axis=0)

prec, recall, f, support = precision_recall_fscore_support(
                                                            golden,
                                                            predict,
                                                            beta=1)

accuracy = accuracy_score(
                            golden,
                            predict
                            )

print('precision:', prec, 'recall:', recall, 'F-score:', f, 'f-avg:', (f[0]+f[1])/2, 'support:', support)
print('accuracy', accuracy)



# output_file = open('vector_space.txt', 'w')
#
# with numpy.printoptions(threshold=numpy.inf):
#     for row in X:
#         for column in X:
#             print(row)
#             output_file.write(str(row))
#             output_file.write('\n')

# numpy.set_printoptions(precision=3)

# l_vectorspace = []
# count = 1
#
# NEW_X = X[0:-1, 0:882014]
# numpy.set_printoptions(threshold=sys.maxsize)
#
# for i in NEW_X:
#     l_vectorspace.append(csr_matrix.toarray(i))
#     # output_file.write(str(csr_matrix.toarray(i)))
#     # output_file.write('\n')
#
# for x in l_vectorspace:
#     output_file.write(str(count))
#     output_file.write('\t')
#     output_file.write(str(x))
#     output_file.write('\n')
#     count = count + 1

# output_file.write(str(l_vectorspace))

# joblib.dump(X, 'vector_space.pkl')
