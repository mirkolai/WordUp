import numpy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import Features_manager
import Database_manager
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold
from itertools import combinations
language="es"
# initialize database_manager
database_manager = Database_manager.make_database_manager(language,"train")
# initialize feature_manager
feature_manager = Features_manager.make_feature_manager()

# recover tweets
instances = numpy.array(database_manager.return_istances())
labels = numpy.array(feature_manager.get_label(instances))

print("istances:", len(instances))

feature_types = feature_manager.get_availablefeaturetypes()
"""
or you could include only desired features"""
feature_types=np.array([
               "ngrams", #1-3 grammi lower binary
               #"boemoji", #1-2 grammi lower binary
               "chargrams", #2-5 chargrammi lower binary
               "puntuactionmarks", #6 feature che contano i più comuni segni di punteggiatura
               "capitalizedletters", #3 feature sull'uso delle maiuscole
               "laughter", #1 featura che conta le risate
               "statistics", #5 feature che verificano la presenza di valori percentuali

               #"bio", #bag of word binary sul testo della bio
               #"cue_words",  #8 feature che contano la presenza di 8 categorie di parole
               #"linguistic_words", #6 feature che contano la presenza di 6 categorie di parole
               "lexical_diversity",#il numero di features varia a seconda della lingua, comprende alcune metriche di complessità linguistic

               "network_centrality_base_retweet",
               "network_centrality_base_friend",
               #"network_centrality_augmented_retweet",
               "network_label_count_base_retweet",
               "network_label_count_base_friend",
               #"network_label_count_augmented_retweet",
               "network_mds_base_retweet",
               "network_mds_base_friend",
               #"network_mds_augmented_retweet",

                #"upos"   ,
                "deprelneg",
                "deprel" ,
                "relationformVERB",
                "relationformNOUN",
                #"relationformADJ",
                "Sidorovbigramsform",
                #"Sidorovbigramsupostag",
                #"Sidorovbigramsdeprel" ,
                "target_context_one", #200 feature che rappresentano gli embeddings della previus e next word rispetto al target nell'albero delle dipendenze
                "target_context_two", #400 feature che rappresentano gli embeddings delle 2 previus e 2 next word rispetto al target nell'albero delle dipendenze
                "tweet_info", #"retweet_count","favourite_count","year","month","hour"
                "tweet_info_source", #one hot encoding sul tipo di media utilizzato per postare il tweet
                "user_info", #"statuses_count","followers_count","friends_count","listed_count","year","month","tweet_posted_at_day"
             ])



# create the feature space with all available features
X, feature_names, feature_type_indexes = feature_manager.create_feature_space(instances, feature_types)
max_f_avg=0
max_feature=None
print("features:", feature_types)
print("feature space dimension:", X.shape)

accuracies=[]
f_avg=[]
print("max_accuracy")
print("max_feature")
print(max_f_avg)
print(max_feature)


"""
https://en.wikipedia.org/wiki/Combination
"""
print("feature space dimension X:", X.shape)

N = len(feature_types)

for K in range(1, N+1):
    for subset in combinations(range(0, N), K):
        print(feature_types[list(subset)])
        feature_index_filtered=numpy.array([list(feature_types).index(f) for f in feature_types[list(subset)]])
        feature_index_filtered=numpy.concatenate(feature_type_indexes[list(feature_index_filtered)])
        X_filter=X[:,feature_index_filtered]
        print("X_filter.shape",X_filter.shape)
        accuracies=[]
        fmacros=[]
        for random_state in [1]:
            kf = KFold(n_splits=5,random_state=random_state,shuffle=True)
            for index_train, index_test in kf.split(X):
                # extract the column of the features considered in the current combination
                # the feature space is reduced
                print("feature space dimension X for ",feature_types[list(subset)],":", X_filter.shape)

                clf= LogisticRegression()

                clf.fit(X_filter[index_train],labels[index_train])
                test_predict = clf.predict(X_filter[index_test])

                prec, recall, f, support = precision_recall_fscore_support(
                    labels[index_test],
                    test_predict,
                    beta=1)

                accuracy = accuracy_score(
                    test_predict,
                    labels[index_test]
                )
                accuracies.append(accuracy)
                fmacros.append((f[0]+f[1])/2)
        print("numpy.mean(accuracy),",numpy.mean(accuracies),accuracies)
        print("numpy.mean(fmacros),",numpy.mean(fmacros),fmacros)
        if(max_f_avg<numpy.mean(fmacros)):
            max_f_avg=numpy.mean(fmacros)
            max_feature=feature_types[list(subset)]
        print("BEST RESULT UNTIL NOW")
        print(max_feature)
        print(max_f_avg)

        file=open("reports/"+language+"_1_2_3_lr_feature_combination.csv","a")
        file.write(', '.join(feature_types[list(subset)])+"\t"+
                   str(X_filter.shape)+"\t"+
                   str(numpy.mean(fmacros))+"\t"+
                   str(numpy.std(fmacros))+"\n")
        file.close()
