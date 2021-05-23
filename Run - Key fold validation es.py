import csv
import sys
from _random import Random

import numpy
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import Features_manager
import Database_manager
import joblib
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold
import random


def unique(list1):
    # intilize a null list
    unique_list = {}

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list[x]=0
        unique_list[x]+=1
            # print list
    for key,value in unique_list.items():
        print("labels:",key,value)


# initialize database_manager
database_manager = Database_manager.make_database_manager("es","train")
# initialize feature_manager
feature_manager = Features_manager.make_feature_manager()

# recover tweets
instances = numpy.array(database_manager.return_istances())
#random.shuffle(instances)
#instances = instances[0:100]
labels = numpy.array(feature_manager.get_label(instances))

print(unique(labels))

print("istances:", len(instances))
"""
ngrams chargrams deprelneg relationformVERB relationformNOUN
ngrams chargrams deprelneg relationformVERB relationformNOUN Sidorovbigramsform
"""

# recover keyword list corresponding to available features
feature_types = feature_manager.get_availablefeaturetypes()
"""
or you could include only desired features"""
feature_types=[
               #"ngrams",
               #"chargrams",
               #"puntuactionmarks",
               #"capitalizedletters",
               #"laughter",

               #"bio",
               #"cue_words",
               #"linguistic_words",

               #"upos"   ,
               #"deprelneg",
               #"deprel" ,
               #"relationformVERB",
               #"relationformNOUN",
               #"relationformADJ",
               #"Sidorovbigramsform",
               #"Sidorovbigramsupostag",
               #"Sidorovbigramsdeprel" ,
               # "target_context_one",
               # "target_context_two",
                "tweet_info",
                "tweet_info_source",
                "user_info",

             ]
#f-avg 0.828633746118431 0.01905251483131286

# create the feature space with all available features
X, feature_names, feature_type_indexes = feature_manager.create_feature_space(instances, feature_types)

print("features:", feature_types)
print("feature space dimension:", X.shape)

accuracies=[]
f_avg=[]
for random_state in [1,2,3]:
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    for index_train, index_test in kf.split(X):
        clf = RandomForestClassifier()
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

