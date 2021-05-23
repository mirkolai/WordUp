import csv
import sys
import numpy
from scipy.sparse import csr_matrix
from sklearn.svm import SVC
import Features_manager
import Database_manager
import joblib
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
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
database_manager = Database_manager.make_database_manager("eu","train")
# initialize feature_manager
feature_manager = Features_manager.make_feature_manager()

# recover tweets
istances = numpy.array(database_manager.return_istances())
labels = numpy.array(feature_manager.get_label(istances))

print(unique(labels))

print("istances:",len(istances))

# recover keyword list corresponding to available features
feature_types = feature_manager.get_availablefeaturetypes()
"""
or you could include only desired features"""
feature_type=[
               "ngrams",
             ]

# create the feature space with all available features
X, feature_names, feature_type_indexes = feature_manager.create_feature_space(istances, feature_types)

print("features:", feature_types)
print("feature space dimension:", X.shape)

accuracies=[]
f_avg=[]
for random_state in [1,2,3]:
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    for index_train, index_test in kf.split(X):
        clf = SVC(kernel="linear")
        clf.fit(X[index_train], labels[index_train])
        test_predict = clf.predict(X[index_test])
        prec, recall, f, support = \
            precision_recall_fscore_support(
                                            labels[index_test],
                                            test_predict,
                                            beta=1)
        accuracy = accuracy_score(
                                    labels[index_test],
                                    test_predict,
                                    )
        accuracies.append((f[0]+f[1])/2)
        f_avg.append((f[0] + f[1]) / 2)
        print('precision:', prec, 'recall:', recall, 'F-score:', f, 'f-avg:', (f[0]+f[1])/2, 'support:', support)
        print('accuracy', accuracy)
print("accuracies",numpy.mean(accuracies),numpy.std(accuracies))
print("f-avg", numpy.mean(f_avg), numpy.std(f_avg))

