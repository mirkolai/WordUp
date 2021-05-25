import csv

import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import Features_manager
import Database_manager

language="eu"

feature_manager = Features_manager.make_feature_manager()

database_manager_train = Database_manager.make_database_manager(language,"train")
instances_train = numpy.array(database_manager_train.return_istances())
labels_train = numpy.array(feature_manager.get_label(instances_train))
database_manager_test = Database_manager.make_database_manager(language,"test")
instances_test = numpy.array(database_manager_test.return_istances())
labels_test = numpy.array(feature_manager.get_label(instances_test))
print("len(instances_train)",len(instances_train))
print("len(instances_test)",len(instances_test))
feature_types = feature_manager.get_availablefeaturetypes()
""" features selezionate per questo track"""
feature_types=[
               "ngrams",
               "chargrams",
               "puntuactionmarks",
               "capitalizedletters",
               "laughter",
               "statistics",

               "bio",
               "cue_words",
               "linguistic_words",
               "lexical_diversity",

               "network_centrality_base_retweet",
               "network_centrality_base_friend",
               "network_centrality_augmented_retweet",
               "network_label_count_base_retweet",
               "network_label_count_base_friend",
               "network_label_count_augmented_retweet",
               "network_mds_base_retweet",
               "network_mds_base_friend",
               "network_mds_augmented_retweet",

               "upos"   ,
               "deprelneg",
               "deprel" ,
               "relationformVERB",
               "relationformNOUN",
               "relationformADJ",
               "Sidorovbigramsform",
               "Sidorovbigramsupostag",
               "Sidorovbigramsdeprel" ,
               "target_context_one",
               "target_context_two",
               "tweet_info",
               "tweet_info_source",
               "user_info",
             ]

X, X_test, feature_name, feature_type_indexes = feature_manager.create_feature_space(instances_train, feature_types,instances_test)

print("feature space dimension X:", X.shape)
print("feature space dimension X_test:", X_test.shape)

clf = RandomForestClassifier()

clf.fit(X, labels_train)
test_predict = clf.predict(X_test)

file=open("predictions/Close-Contextual-track_"+language+"_02.csv","w")
spam_writer= csv.writer(file, delimiter=",",quotechar="\"", quoting=csv.QUOTE_MINIMAL)
spam_writer.writerow(["tweet_id","user_id","text","label"])
for i in range(0, len(test_predict)):
    spam_writer.writerow([instances_test[i].tweet_id,
                          instances_test[i].user_id,
                          instances_test[i].text,
                          test_predict[i]])

