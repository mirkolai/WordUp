import numpy
from numpy.core.multiarray import ndarray
from sklearn.svm.classes import SVC
import Features_manager
import Database_manager
from sklearn.metrics.classification import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold
from itertools import combinations
import csv


def unique(list1):
    # intialize a null list
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
database_manager = Database_manager.make_database_manager()
# initialize feature_manager
feature_manager = Features_manager.make_feature_manager()

# recover tweets
tweets = numpy.array(database_manager.return_tweets_training())
labels = numpy.array(feature_manager.get_label(tweets))

unique(labels)

print("Number of tweets: "+str(len(tweets)))



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
X,feature_names,feature_type_indexes=feature_manager.create_feature_space(tweets,feature_types)


csvfile=open('Main.csv', 'w', newline='')
spamwriter = csv.writer(csvfile, delimiter=',',
                        quotechar='"', quoting=csv.QUOTE_ALL)

spamwriter.writerow(['feature','feature space','prec_0','prec_1','rec_0','rec_1','f_0','f_1','f_avg','support_0','support_1','accuracy'])
csvfile.close()

N = len(feature_types)
for K in range(1, N+1):
    for subset in combinations(range(0, N), K):
        print(subset, N, K)



        feature_index_filtered=numpy.array([list(feature_types).index(f) for f in feature_types[list(subset)]])
        feature_index_filtered=numpy.concatenate(feature_type_indexes[list(feature_index_filtered)])

        print("features:", feature_types[list(subset)])
        print("features names:", feature_names[list(feature_index_filtered)][0:10])
        X_filtered_by_feature=X[:,feature_index_filtered]
        print("feature space dimension:", X_filtered_by_feature.shape)

        golden=[]
        predict=[]
        kf = KFold(n_splits=5, shuffle=True, random_state=True)

        for index_train, index_test in kf.split( X_filtered_by_feature):

            clf = SVC(kernel="linear")

            clf.fit(X_filtered_by_feature[index_train],labels[index_train])
            test_predict = clf.predict(X_filtered_by_feature[index_test])

            golden=numpy.concatenate((golden,labels[index_test]), axis=0)
            predict=numpy.concatenate((predict,test_predict), axis=0)

        prec, recall, f, support = precision_recall_fscore_support(
        golden,
        predict,
        beta=1)

        accuracy = accuracy_score(
        golden,
        predict
        )

        print(prec, recall, f, support )
        print(accuracy)

        csvfile = open('Main.csv', 'a', newline='')
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_ALL)
        spamwriter.writerow(
                    [ ' '.join(feature_types[list(subset)]),
                      X_filtered_by_feature.shape,
                      prec[0], prec[1], recall[0], recall[1], f[0], f[1], (f[0]+f[1])/2,
                      support[0],support[1], accuracy])
        csvfile.close()