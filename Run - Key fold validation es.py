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
               #"ngrams", #1-3 grammi lower binary
               #"chargrams", #2-5 chargrammi lower binary
               #"puntuactionmarks", #6 feature che contano i più comuni segni di punteggiatura
               #"capitalizedletters", #3 feature sull'uso delle maiuscole
               #"laughter", #1 featura che conta le risate

               #"bio", #bag of word binary sul testo della bio
               #"cue_words",  #8 feature che contano la presenza di 8 categorie di parole
               #"linguistic_words", #6 feature che contano la presenza di 6 categorie di parole
               "lexical_diversity",#il numero di features varia aa seconda della lingua, comprende alcune metriche di complessità linguistic
               "statistics", #5 feature che verificano la presenza di valori percentuali
               #"upos"   ,
               #"deprelneg",
               #"deprel" ,
               #"relationformVERB",
               #"relationformNOUN",
               #"relationformADJ",
               #"Sidorovbigramsform",
               #"Sidorovbigramsupostag",
               #"Sidorovbigramsdeprel" ,
               # "target_context_one", #200 feature che rappresentano gli embeddings della previus e next word rispetto al target nell'albero delle dipendenze
               # "target_context_two", #400 feature che rappresentano gli embeddings delle 2 previus e 2 next word rispetto al target nell'albero delle dipendenze
               #"tweet_info", #"retweet_count","favourite_count","year","month","hour"
               #"tweet_info_source", #one hot encoding sul tipo di media utilizzato per postare il tweet
               #"user_info", #"statuses_count","followers_count","friends_count","listed_count","year","month","tweet_posted_at_day"
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

