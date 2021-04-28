from itertools import combinations
import numpy
from sklearn.svm.classes import SVC
import Features_manager
import Database_manager
from sklearn.metrics.classification import precision_recall_fscore_support, accuracy_score

# initialize database_manager
database_manager=Database_manager.make_database_manager()
# initialize feature_manager
feature_manager=Features_manager.make_feature_manager()

# recover training tweets
tweets_training=numpy.array(database_manager.return_tweets_training())
labels_training=numpy.array(feature_manager.get_label(tweets_training))

# recover test tweets
tweets_test=numpy.array(database_manager.return_tweets_test())
labels_test=numpy.array(feature_manager.get_label(tweets_test))

# recover keyword list corresponding to available features
feature_types=feature_manager.get_availablefeaturetypes()

# create the feature space with all available features
X,X_test,feature_names,feature_type_indexes=feature_manager.create_feature_space(tweets_training,feature_types,tweets_test)

print("feature space dimension X:", X.shape)
print("feature space dimension X_test:", X_test.shape)
"""
https://en.wikipedia.org/wiki/Combination
"""
N = len(feature_types)
for K in range(1, N):
    for subset in combinations(range(1, N), K):


        feature_index_filtered=numpy.array([list(feature_types).index(f) for f in feature_types[list(subset)]])
        feature_index_filtered=numpy.concatenate(feature_type_indexes[list(feature_index_filtered)])

        # extract the column of the features considered in the current combination
        # the feature space is reduced
        X_filter=X[:,feature_index_filtered]
        X_test_filter=X_test[:,feature_index_filtered]

        clf = SVC(kernel='linear')

        clf.fit(X_filter, labels_training)
        test_predict = clf.predict(X_test_filter)



        prec, recall, f, support = precision_recall_fscore_support(
        labels_test,
        test_predict,
        beta=1)

        accuracy = accuracy_score(
        test_predict,
        labels_test
        )

        print(feature_types[list(subset)])
        print("feature space dimention X:", X_filter.shape)
        print("feature space dimention X_Test:", X_test_filter.shape)
        print(prec, recall, f, support)
        print(accuracy)


