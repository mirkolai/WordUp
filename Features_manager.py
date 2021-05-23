from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
from scipy.sparse import csr_matrix, hstack
from datetime import datetime
from resource_lexical_cue_words import Cue_Words
from resource_lexical_linguistic_words import Linguistic_Words


class Features_manager(object):

    def __init__(self):
        """You could add new feature types in global_feature_types_list
            global_feature_types_list is  a dictionary containing the feature space matrix for each feature type

            if you want to add a new feature:
            1. chose a keyword for defining the  feature type
            2. define a function function_name(self,tweets,tweet_test=None) where:

                tweets: Array of  tweet objects belonging to the training set
                tweet_test: Optional. Array of tweet objects belonging to the test set

                return:
                X_train: The feature space of the training set (numpy.array)
                X_test: The feature space of the test set, if test  set was defined (numpy.array)
                feature_names: An array containing the names of the features used for creating the feature space (array)

        """
        self.global_feature_types_list={
             "ngrams"               :self.get_ngrams_features,
             "chargrams"            :self.get_nchargrams_features,
             "puntuactionmarks"     :self.get_puntuaction_marks_features,
             "capitalizedletters"   :self.get_capitalized_letters_features,
             "laughter"             :self.get_laughter_features,
             "statistics"           :self.get_statistics_features,

             "cue_words"             :self.get_cue_words_features,
             "linguistic_words"      :self.get_linguistic_words_features,
             "lexical_diversity"     :self.get_lexical_diversity_features,

             "network_centrality_base_retweet"      :self.get_networks_metrics_base_centrality_retweet_features,
             "network_centrality_base_friend"       :self.get_networks_metrics_base_centrality_friend_features,
             "network_centrality_augmented_retweet" :self.get_networks_metrics_augmented_centrality_retweet_features,

             "network_label_count_base_retweet"      :self.get_networks_metrics_base_label_count_retweet_features,
             "network_label_count_base_friend"       :self.get_networks_metrics_base_label_count_friend_features,
             "network_label_count_augmented_retweet" :self.get_networks_metrics_augmented_label_count_retweet_features,

              "bio"                 :self.get_bow_bio_features,

             "upos"                 :self.get_upos_features,
             "deprelneg"            :self.get_deprelneg_features,
             "deprel"               :self.get_deprel_features,
             "relationformVERB"     :self.get_relationformVERB_features,
             "relationformNOUN"     :self.get_relationformNOUN_features,
             "relationformADJ"      :self.get_relationformADJ_features,
             "Sidorovbigramsform"   :self.get_Sidorov_bigramsform_features,
             "Sidorovbigramsupostag":self.get_Sidorov_bigramsupostag_features,
             "Sidorovbigramsdeprel" :self.get_Sidorov_bigramsdeprel_features,

             "target_context_one"   :self.get_target_context_one_features,
             "target_context_two"   :self.get_target_context_two_features,

             "tweet_info"            :self.get_tweet_info_features,
             "user_info"             :self.get_user_info_features,
             "tweet_info_source"     :self.get_tweet_info_source_features,

        }

        return

    def get_availablefeaturetypes(self):
        """
        Return un array containing the keyword corresponding to  available feature types
        :return: un array containing the keyword corresponding to  available feature types
        """
        return np.array([ x for x in self.global_feature_types_list.keys()])

    def get_label(self,tweets):
        """
        Return un array containing the label for each tweet
        :param tweets:  Array of Tweet Objects
        :return: Array of label
        """
        return [ tweet.label for tweet  in tweets]


    #features extractor
    def create_feature_space(self,tweets,feature_types_list=None,tweet_test=None):
        """

        :param tweets: Array of  tweet objects belonging to the training set
        :param feature_types_list: Optional. array of keyword corresponding to global_feature_types_list (accepted  values are:
            "ngrams"
            "ngramshashtag"
            "chargrams"
            "numhashtag"
            "puntuactionmarks"
            "length")
            If not defined, all available features are used


            You could add new features in global_feature_types_list. See def __init__(self).

            
        :param tweet_test: Optional. Array of tweet objects belonging to the test set
        :return:
        
        X: The feature space of the training set
        X_test: The feature space of the test set, if test  set was defined
        feature_names: An array containing the names of the features used  for  creating the feature space
        feature_type_indexes: A numpy array of length len(feature_type).
                           Given feature_type i, feature_type_indexes[i] contains
                           the list of the index columns of the feature space matrix for the feature type i

        How to use the output, example:

        feature_type=feature_manager.get_avaiblefeaturetypes()

        print(feature_type)  # available feature types
        [
          "puntuactionmarks",
          "length",
          "numhashtag",
        ]
        print(feature_name)  # the name of all feature corresponding to the  number of columns of X
        ['feature_exclamation', 'feature_question',
        'feature_period', 'feature_comma', 'feature_semicolon','feature_overall',
        'feature_charlen', 'feature_avgwordleng', 'feature_numword',
        'feature_numhashtag' ]

        print(feature_type_indexes)
        [ [0,1,2,3,4,5],
          [6,7,8],
          [9]
        ]

        print(X) #feature space 3X10 using "puntuactionmarks", "length", and "numhashtag"
        numpy.array([
        [0,1,0,0,0,1,1,0,0,1], # vector rapresentation of the document 1
        [0,1,1,1,0,1,1,0,0,1], # vector rapresentation of the document 2
        [0,1,0,1,0,1,0,1,1,1], # vector rapresentation of the document 3

        ])

        # feature space 3X6 obtained using only "puntuactionmarks"
        print(X[:, feature_type_indexes[feature_type.index("puntuactionmarks")]])
        numpy.array([
        [0,1,0,0,0,1], # vector representation of the document 1
        [0,1,1,1,0,1], # vector representation of the document 2
        [0,1,0,1,0,1], # vector representation of the document 3

        ])

        """

        if feature_types_list is None:
            feature_types_list=self.get_availablefeaturetypes()



        if tweet_test is None:
            all_feature_names=[]
            all_feature_index=[]
            all_X=[]
            index=0
            for key in feature_types_list:
                X,feature_names=self.global_feature_types_list[key](tweets,tweet_test)

                current_feature_index=[]
                for i in range(0,len(feature_names)):
                    current_feature_index.append(index)
                    index+=1
                all_feature_index.append(current_feature_index)

                all_feature_names=np.concatenate((all_feature_names,feature_names))
                if all_X!=[]:
                    all_X=csr_matrix(hstack((all_X,X)))
                else:
                    all_X=X

            return all_X, all_feature_names, np.array(all_feature_index)
        else:
            all_feature_names=[]
            all_feature_index=[]
            all_X=[]
            all_X_test=[]
            index=0
            for key in feature_types_list:
                X,X_test,feature_names=self.global_feature_types_list[key](tweets,tweet_test)

                current_feature_index=[]
                for i in range(0,len(feature_names)):
                    current_feature_index.append(index)
                    index+=1
                all_feature_index.append(current_feature_index)

                all_feature_names=np.concatenate((all_feature_names,feature_names))
                if all_X!=[]:
                    all_X=csr_matrix(hstack((all_X,X)))
                    all_X_test=csr_matrix(hstack((all_X_test,X_test)))
                else:
                    all_X=X
                    all_X_test=X_test

            return all_X, all_X_test, all_feature_names, np.array(all_feature_index)

#################################
# Structural
#################################

    def get_ngrams_features(self, tweets,tweet_test=None):


        tfidfVectorizer = CountVectorizer(ngram_range=(1,3),
                                          analyzer="word",
                                          #stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)

        if tweet_test is None:
            feature = []
            for tweet in tweets:

                feature.append(tweet.text)


            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names=tfidfVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature  = []
            feature_test  = []
            for tweet in tweets:

                feature.append(tweet.text)

            for tweet in tweet_test:

                feature_test.append(tweet.text)


            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X_train = tfidfVectorizer.transform(feature)
            X_test = tfidfVectorizer.transform(feature_test)

            feature_names=tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

    def get_nchargrams_features(self, tweets,tweet_test=None):


        # CountVectorizer return a numpy matrix
        # row number of tweets
        # column number of 2-5chargrams in the dictionary
        tfidfVectorizer = CountVectorizer(ngram_range=(2, 5),
                                          analyzer="char",
                                          #stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)

        if tweet_test is None:
            feature = []
            for tweet in tweets:

                feature.append(tweet.text)


            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names=tfidfVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:

                feature.append(tweet.text)

            for tweet in tweet_test:

                feature_test.append(tweet.text)


            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X_train = tfidfVectorizer.transform(feature)
            X_test = tfidfVectorizer.transform(feature_test)

            feature_names=tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

    def get_bow_bio_features(self, tweets,tweet_test=None):


        tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                          analyzer="word",
                                          #stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)

        if tweet_test is None:
            feature = []
            for tweet in tweets:
                 feature.append(tweet.bio)


            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names=tfidfVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature  = []
            feature_test  = []
            for tweet in tweets:

                feature.append(tweet.bio)

            for tweet in tweet_test:

                feature_test.append(tweet.bio)


            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X_train = tfidfVectorizer.transform(feature)
            X_test = tfidfVectorizer.transform(feature_test)

            feature_names=tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

#################################
# Structural 2
#################################

    def get_puntuaction_marks_features(self,tweets,tweet_test=None):

        # This method extracts a single column (feature_numhashtag)
        # len(tweets) rows of 6 columns
        # sr_matrix(np.vstack(feature)) convert to an array of dimension len(tweets)X6

        if tweet_test is None:
            feature = []

            for tweet in tweets:
                feature.append([
                len(re.findall(r"[!]", tweet.text)),
                len(re.findall(r"[?]", tweet.text)),
                len(re.findall(r"[.]", tweet.text)),
                len(re.findall(r"[,]", tweet.text)),
                len(re.findall(r"[;]", tweet.text)),
                len(re.findall(r"[!?.,;]", tweet.text)),
                ]

            )


            return csr_matrix(np.vstack(feature)),\
                   ["feature_exclamation",
                    "feature_question",
                    "feature_period",
                    "feature_comma",
                    "feature_semicolon",
                    "feature_overall"]


        else:
            feature  = []
            feature_test  = []

            for tweet in tweets:
                feature.append([
                len(re.findall(r"[!]", tweet.text)),
                len(re.findall(r"[?]", tweet.text)),
                len(re.findall(r"[.]", tweet.text)),
                len(re.findall(r"[,]", tweet.text)),
                len(re.findall(r"[;]", tweet.text)),
                len(re.findall(r"[!?.,;]", tweet.text)),
                ]

            )


            for tweet in tweet_test:
                feature_test.append([
                len(re.findall(r"[!]", tweet.text)),
                len(re.findall(r"[?]", tweet.text)),
                len(re.findall(r"[.]", tweet.text)),
                len(re.findall(r"[,]", tweet.text)),
                len(re.findall(r"[;]", tweet.text)),
                len(re.findall(r"[!?.,;]", tweet.text)),
                ]

            )


            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_test)),\
                   ["feature_exclamation",
                    "feature_question",
                    "feature_period",
                    "feature_comma",
                    "feature_semicolon",
                    "feature_overall"]

    def get_capitalized_letters_features(self,tweets,tweet_test=None):


        if tweet_test is None:
            feature  = []
            for tweet in tweets:
                feature.append([
                len(re.findall(r"[:upper:]{2,}", tweet.text)),
                len(re.findall(r"[:upper:][:lower:]{1,}", tweet.text)),
                len(re.findall(r"[:lower:]{1,}[:upper:]{1,}[:lower:]{1,}", tweet.text)),
                ]

            )


            return csr_matrix(np.vstack(feature)),\
                   ["feature_words_all_capital",
                    "feature_words_start_with_capital",
                    "feature_words_with_a_capital_letter_in_the_middle",]


        else:
            feature  = []
            feature_test  = []

            for tweet in tweets:
                feature.append([
                len(re.findall(r"[:upper:]{2,}", tweet.text)),
                len(re.findall(r"[:upper:][:lower:]{1,}", tweet.text)),
                len(re.findall(r"[:lower:]{1,}[:upper:]{1,}[:lower:]{1,}", tweet.text)),
                ]

            )

            for tweet in tweet_test:
                feature_test.append([
                len(re.findall(r"[:upper:]{2,}", tweet.text)),
                len(re.findall(r"[:upper:][:lower:]{1,}", tweet.text)),
                len(re.findall(r"[:lower:]{1,}[:upper:]{1,}[:lower:]{1,}", tweet.text)),
                ]

            )

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_test)),\
                   ["feature_words_all_capital",
                    "feature_words_start_with_capital",
                    "feature_words_with_a_capital_letter_in_the_middle",]

    def get_laughter_features(self,tweets,tweet_test=None):


        if tweet_test is None:
            feature = []
            #migliorare
            for tweet in tweets:
                feature.append([
                len(re.findall(r"((ah[ ]{0,}){2,}|(eh[ ]{0,}){2,}|(ih[ ]{0,}){2,}|(ja[ ]{0,}){2,}|(je[ ]{0,}){2,}|(ji[ ]{0,}){2,})", tweet.text))]

            )

            return csr_matrix(np.vstack(feature)),\
                   ["feature_laughter"]


        else:
            feature  = []
            feature_test  = []

            for tweet in tweets:
                feature.append([
                    len(re.findall(r"((ah){2,}|(eh){2,}|(ih){2,}|(ja){2,}|(je){2,}|(ji){2,})", tweet.text))]

                )


            for tweet in tweet_test:
                feature_test.append([
                    len(re.findall(r"((ah){2,}|(eh){2,}|(ih){2,}|(ja){2,}|(je){2,}|(ji){2,})", tweet.text))]

                )

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_test)),\
                   ["feature_laughter"]

    def get_statistics_features(self,tweets,tweet_test=None):

        # This method extracts a single column (feature_numhashtag)
        # len(tweets) rows of 6 columns
        # sr_matrix(np.vstack(feature)) convert to an array of dimension len(tweets)X6

        if tweet_test is None:
            feature = []

            for tweet in tweets:
                feature.append([
                len(re.findall(r"[\%]", tweet.text)),
                len(re.findall(r"[0-9\.]{1,}\%", tweet.text)),
                len(re.findall(r"[^0-9][5-9]{1}[0-9]{1}(\.[0-9]{0,})?\%", tweet.text)),
                len(re.findall(r"[^0-9][0-4]{1}[0-9]{1}(\.[0-9]{0,})?\%", tweet.text)),
                len(re.findall(r"[^0-9][9]{1}[0-9]{1}(\.[0-9]{0,})?\%", tweet.text)),
                ]

            )


            return csr_matrix(np.vstack(feature)),\
                   ["feature_percentage",
                    "feature_numbers_percentage",
                    "feature_pergentage_more_50",
                    "feature_pergentage_les_50",
                    "feature_pergentage_more_90"]


        else:
            feature  = []
            feature_test  = []

            for tweet in tweets:
                feature.append([
                len(re.findall(r"[\%]", tweet.text)),
                len(re.findall(r"[0-9\.]{1,}\%", tweet.text)),
                len(re.findall(r"[^0-9][5-9]{1}[0-9]{1}(\.[0-9]{0,})?\%", tweet.text)),
                len(re.findall(r"[^0-9][0-4]{1}[0-9]{1}(\.[0-9]{0,})?\%", tweet.text)),
                len(re.findall(r"[^0-9][9]{1}[0-9]{1}(\.[0-9]{0,})?\%", tweet.text)),
                ]

            )


            for tweet in tweet_test:
                feature_test.append([
                len(re.findall(r"[\%]", tweet.text)),
                len(re.findall(r"[0-9\.]{1,}\%", tweet.text)),
                len(re.findall(r"[^0-9][5-9]{1}[0-9]{1}(\.[0-9]{0,})?\%", tweet.text)),
                len(re.findall(r"[^0-9][0-4]{1}[0-9]{1}(\.[0-9]{0,})?\%", tweet.text)),
                len(re.findall(r"[^0-9][9]{1}[0-9]{1}(\.[0-9]{0,})?\%", tweet.text)),
                ]

            )


            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_test)),\
                   ["feature_percentage",
                    "feature_numbers_percentage",
                    "feature_pergentage_more_50",
                    "feature_pergentage_les_50",
                    "feature_pergentage_more_90"]


#################################
#lessicals
################################

    def get_cue_words_features(self, tweets, tweets_test=None):
        print("Calculating cue_words feature...")
        model = Cue_Words(tweets[0].language)

        if tweets_test is None:
            feature = []
            for tweet in tweets:
                concepts, values = model.get_feature(tweet)
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                concepts, values = model.get_feature(tweet)
                feature.append(values)

            feature_names = concepts

            for tweet in tweets:
                concepts, values = model.get_feature(tweet)
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names

    def get_linguistic_words_features(self, tweets, tweets_test=None):
        print("Calculating linguistic_words feature...")
        model = Linguistic_Words(tweets[0].language)

        if tweets_test is None:
            feature = []
            for tweet in tweets:
                concepts, values = model.get_feature(tweet)
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                concepts, values = model.get_feature(tweet)
                feature.append(values)

            feature_names = concepts

            for tweet in tweets:
                concepts, values = model.get_feature(tweet)
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names


    def get_lexical_diversity_features(self, tweets, tweets_test=None):
        print("Calculating lexical_diversity feature...")
        if tweets_test is None:
            feature = []
            for tweet in tweets:
                concepts = [key   for key,value in tweet.lexical_diversity.dimensions.items()]
                values   = [float(value) for key,value in tweet.lexical_diversity.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                concepts = [key   for key,value in tweet.lexical_diversity.dimensions.items()]
                values   = [float(value) for key,value in tweet.lexical_diversity.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            for tweet in tweets:
                concepts = [key   for key,value in tweet.lexical_diversity.dimensions.items()]
                values   = [float(value) for key,value in tweet.lexical_diversity.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names

##############################################
##networks
##################################################

    def get_networks_metrics_base_centrality_retweet_features(self, tweets, tweets_test=None):
        print("Calculating lexical_diversity feature...")
        if tweets_test is None:
            feature = []
            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_base_centrality_retweet.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_base_centrality_retweet.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_base_centrality_retweet.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_base_centrality_retweet.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_base_centrality_retweet.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_base_centrality_retweet.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names

    def get_networks_metrics_base_centrality_friend_features(self, tweets, tweets_test=None):
        print("Calculating lexical_diversity feature...")
        if tweets_test is None:
            feature = []
            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_base_centrality_friend.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_base_centrality_friend.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_base_centrality_friend.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_base_centrality_friend.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_base_centrality_friend.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_base_centrality_friend.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names

    def get_networks_metrics_augmented_centrality_retweet_features(self, tweets, tweets_test=None):
        print("Calculating lexical_diversity feature...")
        if tweets_test is None:
            feature = []
            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_augmented_centrality_retweet.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_augmented_centrality_retweet.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_augmented_centrality_retweet.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_augmented_centrality_retweet.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_augmented_centrality_retweet.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_augmented_centrality_retweet.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names


    def get_networks_metrics_base_label_count_retweet_features(self, tweets, tweets_test=None):
        print("Calculating lexical_diversity feature...")
        if tweets_test is None:
            feature = []
            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_base_label_count_retweet.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_base_label_count_retweet.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_base_label_count_retweet.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_base_label_count_retweet.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_base_label_count_retweet.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_base_label_count_retweet.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names

    def get_networks_metrics_base_label_count_friend_features(self, tweets, tweets_test=None):
        print("Calculating lexical_diversity feature...")
        if tweets_test is None:
            feature = []
            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_base_label_count_friend.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_base_label_count_friend.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_base_label_count_friend.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_base_label_count_friend.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_base_label_count_friend.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_base_label_count_friend.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names

    def get_networks_metrics_augmented_label_count_retweet_features(self, tweets, tweets_test=None):
        print("Calculating lexical_diversity feature...")
        if tweets_test is None:
            feature = []
            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_augmented_label_count_retweet.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_augmented_label_count_retweet.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_augmented_label_count_retweet.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_augmented_label_count_retweet.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            for tweet in tweets:
                concepts = [key   for key,value in tweet.networks_metrics_augmented_label_count_retweet.dimensions.items()]
                values   = [float(value) for key,value in tweet.networks_metrics_augmented_label_count_retweet.dimensions.items()]
                feature.append(values)

            feature_names = concepts

            return csr_matrix(np.vstack(feature)), csr_matrix(np.vstack(feature_test)), feature_names





#################################
# Udpipe
#################################
    def get_upos_features(self, tweets, tweet_test=None):
        """

        :param tweets: Array of  Tweet objects. Training set.
        :param tweet_test: Optional Array of  Tweet objects. Test set.
        :return:

        X_train: The feature space of the training set
        X_test: The feature space of the test set, if test  set was defined
        feature_names:  An array containing the names of the features used  for  creating the feature space
        """
        # CountVectorizer return a numpy matrix
        # row number of tweets
        # column number of 1-3gram in the dictionary

        tfidfVectorizer = CountVectorizer(ngram_range=(3, 6),
                                          analyzer="word",
                                          # stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)

        if tweet_test is None:
            feature = []
            for tweet in tweets:
                feature.append(tweet.upostag)

            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names = tfidfVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                feature.append(tweet.upostag)

            for tweet in tweet_test:
                feature_test.append(tweet.upostag)

            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X_train = tfidfVectorizer.transform(feature)
            X_test = tfidfVectorizer.transform(feature_test)

            feature_names = tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

    def get_deprelneg_features(self, tweets, tweet_test=None):
        """

        :param tweets: Array of  Tweet objects. Training set.
        :param tweet_test: Optional Array of  Tweet objects. Test set.
        :return:

        X_train: The feature space of the training set
        X_test: The feature space of the test set, if test  set was defined
        feature_names:  An array containing the names of the features used  for  creating the feature space
        """
        # CountVectorizer return a numpy matrix
        # row number of tweets
        # column number of 1-3gram in the dictionary

        tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                          analyzer="word",
                                          # stop_words="english",
                                          lowercase=True,
                                          binary=False,
                                          max_features=500000)

        if tweet_test is None:
            feature = []
            for tweet in tweets:
                feature.append(tweet.deprelnegation)


            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names = tfidfVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                feature.append(tweet.deprelnegation)

            for tweet in tweet_test:
                feature_test.append(tweet.deprelnegation)

            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X_train = tfidfVectorizer.transform(feature)
            X_test = tfidfVectorizer.transform(feature_test)

            feature_names = tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

    def get_deprel_features(self, tweets, tweet_test=None):
        """

        :param tweets: Array of  Tweet objects. Training set.
        :param tweet_test: Optional Array of  Tweet objects. Test set.
        :return:

        X_train: The feature space of the training set
        X_test: The feature space of the test set, if test  set was defined
        feature_names:  An array containing the names of the features used  for  creating the feature space
        """
        # CountVectorizer return a numpy matrix
        # row number of tweets
        # column number of 1-3gram in the dictionary

        tfidfVectorizer = CountVectorizer(ngram_range=(5,7),
                                          analyzer="word",
                                          # stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)

        if tweet_test is None:
            feature = []
            for tweet in tweets:
                feature.append(tweet.deprels)


            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names = tfidfVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                feature.append(tweet.deprels)

            for tweet in tweet_test:
                feature_test.append(tweet.deprels)

            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X_train = tfidfVectorizer.transform(feature)
            X_test = tfidfVectorizer.transform(feature_test)

            feature_names = tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

    def get_relationformVERB_features(self, tweets, tweet_test=None):
        """

        :param tweets: Array of  Tweet objects. Training set.
        :param tweet_test: Optional Array of  Tweet objects. Test set.
        :return:

        X_train: The feature space of the training set
        X_test: The feature space of the test set, if test  set was defined
        feature_names:  An array containing the names of the features used  for  creating the feature space
        """
        # CountVectorizer return a numpy matrix
        # row number of tweets
        # column number of 1-3gram in the dictionary

        tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                          analyzer="word",
                                          # stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)

        if tweet_test is None:
            feature = []
            for tweet in tweets:
                feature.append(tweet.relationVERB)


            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names = tfidfVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                feature.append(tweet.relationVERB)

            for tweet in tweet_test:
                feature_test.append(tweet.relationVERB)

            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X_train = tfidfVectorizer.transform(feature)
            X_test = tfidfVectorizer.transform(feature_test)

            feature_names = tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

    def get_relationformNOUN_features(self, tweets, tweet_test=None):
        """

        :param tweets: Array of  Tweet objects. Training set.
        :param tweet_test: Optional Array of  Tweet objects. Test set.
        :return:

        X_train: The feature space of the training set
        X_test: The feature space of the test set, if test  set was defined
        feature_names:  An array containing the names of the features used  for  creating the feature space
        """
        # CountVectorizer return a numpy matrix
        # row number of tweets
        # column number of 1-3gram in the dictionary

        tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                          analyzer="word",
                                          # stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)

        if tweet_test is None:
            feature = []
            for tweet in tweets:
                feature.append(tweet.relationNOUN)


            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names = tfidfVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                feature.append(tweet.relationNOUN)

            for tweet in tweet_test:
                feature_test.append(tweet.relationNOUN)

            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X_train = tfidfVectorizer.transform(feature)
            X_test = tfidfVectorizer.transform(feature_test)

            feature_names = tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

    def get_relationformADJ_features(self, tweets, tweet_test=None):
        """

        :param tweets: Array of  Tweet objects. Training set.
        :param tweet_test: Optional Array of  Tweet objects. Test set.
        :return:

        X_train: The feature space of the training set
        X_test: The feature space of the test set, if test  set was defined
        feature_names:  An array containing the names of the features used  for  creating the feature space
        """
        # CountVectorizer return a numpy matrix
        # row number of tweets
        # column number of 1-3gram in the dictionary

        tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                          analyzer="word",
                                          # stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)

        if tweet_test is None:
            feature = []
            for tweet in tweets:
                feature.append(tweet.relationADJ)


            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names = tfidfVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                feature.append(tweet.relationADJ)

            for tweet in tweet_test:
                feature_test.append(tweet.relationADJ)

            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X_train = tfidfVectorizer.transform(feature)
            X_test = tfidfVectorizer.transform(feature_test)

            feature_names = tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

    def get_Sidorov_bigramsform_features(self, tweets, tweet_test=None):
        """

        :param tweets: Array of  Tweet objects. Training set.
        :param tweet_test: Optional Array of  Tweet objects. Test set.
        :return:

        X_train: The feature space of the training set
        X_test: The feature space of the test set, if test  set was defined
        feature_names:  An array containing the names of the features used  for  creating the feature space
        """
        # CountVectorizer return a numpy matrix
        # row number of tweets
        # column number of 1-3gram in the dictionary

        tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                          analyzer="word",
                                          # stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)

        if tweet_test is None:
            feature = []
            for tweet in tweets:
                feature.append(tweet.Sidorov_form)


            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names = tfidfVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                feature.append(tweet.Sidorov_form)

            for tweet in tweet_test:
                feature_test.append(tweet.Sidorov_form)

            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X_train = tfidfVectorizer.transform(feature)
            X_test = tfidfVectorizer.transform(feature_test)

            feature_names = tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

    def get_Sidorov_bigramsupostag_features(self, tweets, tweet_test=None):
        """

        :param tweets: Array of  Tweet objects. Training set.
        :param tweet_test: Optional Array of  Tweet objects. Test set.
        :return:

        X_train: The feature space of the training set
        X_test: The feature space of the test set, if test  set was defined
        feature_names:  An array containing the names of the features used  for  creating the feature space
        """
        # CountVectorizer return a numpy matrix
        # row number of tweets
        # column number of 1-3gram in the dictionary

        tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                          analyzer="word",
                                          # stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)

        if tweet_test is None:
            feature = []
            for tweet in tweets:
                feature.append(tweet.Sidorov_upostag)


            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names = tfidfVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                feature.append(tweet.Sidorov_upostag)

            for tweet in tweet_test:
                feature_test.append(tweet.Sidorov_upostag)

            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X_train = tfidfVectorizer.transform(feature)
            X_test = tfidfVectorizer.transform(feature_test)

            feature_names = tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names

    def get_Sidorov_bigramsdeprel_features(self, tweets, tweet_test=None):
            """

            :param tweets: Array of  Tweet objects. Training set.
            :param tweet_test: Optional Array of  Tweet objects. Test set.
            :return:

            X_train: The feature space of the training set
            X_test: The feature space of the test set, if test  set was defined
            feature_names:  An array containing the names of the features used  for  creating the feature space
            """
            # CountVectorizer return a numpy matrix
            # row number of tweets
            # column number of 1-3gram in the dictionary

            tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                              analyzer="word",
                                              # stop_words="english",
                                              lowercase=True,
                                              binary=True,
                                              max_features=500000)

            if tweet_test is None:
                feature = []
                for tweet in tweets:
                    feature.append(tweet.Sidorov_deprel)

                tfidfVectorizer = tfidfVectorizer.fit(feature)

                X = tfidfVectorizer.transform(feature)

                feature_names = tfidfVectorizer.get_feature_names()

                return X, feature_names
            else:
                feature = []
                feature_test = []
                for tweet in tweets:
                    feature.append(tweet.Sidorov_deprel)

                for tweet in tweet_test:
                    feature_test.append(tweet.Sidorov_deprel)

                tfidfVectorizer = tfidfVectorizer.fit(feature)

                X_train = tfidfVectorizer.transform(feature)
                X_test = tfidfVectorizer.transform(feature_test)

                feature_names = tfidfVectorizer.get_feature_names()

                return X_train, X_test, feature_names


    def get_target_context_one_features(self, tweets, tweets_test=None):

        if tweets_test is None:
            feature = []
            for tweet in tweets:
                values = tweet.target_context_one
                feature.append(values)

            return csr_matrix(np.vstack(feature)),\
                   [str(i)+"_target_context_one" for i in range(0,len(values)) ]
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                concepts, values = tweet.target_context_one
                feature.append(values)

            for tweet in tweets:
                concepts, values = tweet.target_context_one
                feature.append(values)

            return csr_matrix(np.vstack(feature)),\
                   csr_matrix(np.vstack(feature_test)),\
                   [str(i)+"_target_context_one" for i in range(0,len(values)) ]


    def get_target_context_two_features(self, tweets, tweets_test=None):

        if tweets_test is None:
            feature = []
            for tweet in tweets:
                values = tweet.target_context_one
                feature.append(values)

            return csr_matrix(np.vstack(feature)),\
                   [str(i)+"_target_context_one" for i in range(0,len(values)) ]
        else:
            feature = []
            feature_test = []
            for tweet in tweets:
                concepts, values = tweet.target_context_one
                feature.append(values)

            for tweet in tweets:
                concepts, values = tweet.target_context_one
                feature.append(values)

            return csr_matrix(np.vstack(feature)),\
                   csr_matrix(np.vstack(feature_test)),\
                   [str(i)+"_target_context_one" for i in range(0,len(values)) ]




####################################
#base features
#####################################

    def get_tweet_info_features(self, tweets,tweet_test=None):



        if tweet_test is None:
            feature = []
            for tweet in tweets:
                feature.append([
                    int(tweet.tweet.retweet_count),
                    int(tweet.tweet.favourite_count),
                    datetime.strptime(tweet.tweet.created_at, '%Y-%m-%d %H:%M:%S').year,
                    datetime.strptime(tweet.tweet.created_at, '%Y-%m-%d %H:%M:%S').month,
                    datetime.strptime(tweet.tweet.created_at, '%Y-%m-%d %H:%M:%S').hour,
                ])

            return csr_matrix(np.vstack(feature)),\
                   ["retweet_count",
                    "favourite_count",
                    "year",
                    "month",
                    "hour"]
        else:
            feature  = []
            feature_test  = []

            for tweet in tweets:
                feature.append([
                    int(tweet.tweet.retweet_count),
                    int(tweet.tweet.favourite_count),
                    datetime.strptime(tweet.tweet.created_at, '%Y-%m-%d %H:%M:%S').year,
                    datetime.strptime(tweet.tweet.created_at, '%Y-%m-%d %H:%M:%S').month,
                    datetime.strptime(tweet.tweet.created_at, '%Y-%m-%d %H:%M:%S').hour,
                ])


            for tweet in tweet_test:
                feature_test.append([
                    int(tweet.tweet.retweet_count),
                    int(tweet.tweet.favourite_count),
                    datetime.strptime(tweet.tweet.created_at, '%Y-%m-%d %H:%M:%S').year,
                    datetime.strptime(tweet.tweet.created_at, '%Y-%m-%d %H:%M:%S').month,
                    datetime.strptime(tweet.tweet.created_at, '%Y-%m-%d %H:%M:%S').hour,
                ])

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_test)),\
                   ["retweet_count",
                    "favourite_count",
                    "year",
                    "month",
                    "hour"]


    def get_user_info_features(self, tweets,tweet_test=None):


        #,created_at
        if tweet_test is None:
            feature = []
            for tweet in tweets:
                feature.append([
                    int(tweet.user.statuses_count),
                    int(tweet.user.followers_count),
                    int(tweet.user.friends_count),
                    int(tweet.user.listed_count),
                    datetime.strptime(tweet.user.created_at, '%Y-%m-%d %H:%M:%S').year,
                    datetime.strptime(tweet.user.created_at, '%Y-%m-%d %H:%M:%S').month,
                    int(tweet.user.statuses_count)/((datetime.strptime(tweet.tweet.created_at, '%Y-%m-%d %H:%M:%S')-datetime.strptime(tweet.user.created_at, '%Y-%m-%d %H:%M:%S')).days+0.1)
                ])

            return csr_matrix(np.vstack(feature)),\
                   ["statuses_count",
                    "followers_count",
                    "friends_count",
                    "listed_count",
                    "year",
                    "month",
                    "tweet_posted_at_day"]
        else:
            feature  = []
            feature_test  = []

            for tweet in tweets:
                feature.append([
                    int(tweet.user.statuses_count),
                    int(tweet.user.followers_count),
                    int(tweet.user.friends_count),
                    int(tweet.user.listed_count),
                    datetime.strptime(tweet.user.created_at, '%Y-%m-%d %H:%M:%S').year,
                    datetime.strptime(tweet.created_at, '%Y-%m-%d %H:%M:%S').month,
                    int(tweet.user.statuses_count)/((datetime.strptime(tweet.tweet.created_at, '%Y-%m-%d %H:%M:%S')-datetime.strptime(tweet.user.created_at, '%Y-%m-%d %H:%M:%S')).days+0.1)
                ])


            for tweet in tweet_test:
                feature_test.append([
                    int(tweet.user.statuses_count),
                    int(tweet.user.followers_count),
                    int(tweet.user.friends_count),
                    int(tweet.user.listed_count),
                    datetime.strptime(tweet.user.created_at, '%Y-%m-%d %H:%M:%S').year,
                    datetime.strptime(tweet.created_at, '%Y-%m-%d %H:%M:%S').month,
                    int(tweet.user.statuses_count)/((datetime.strptime(tweet.tweet.created_at, '%Y-%m-%d %H:%M:%S')-datetime.strptime(tweet.user.created_at, '%Y-%m-%d %H:%M:%S')).days+0.1)
                ])

            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_test)),\
                   ["statuses_count",
                    "followers_count",
                    "friends_count",
                    "listed_count",
                    "year",
                    "month",
                    "tweet_posted_at_day"]

    def get_tweet_info_source_features(self, tweets,tweet_test=None):


        tfidfVectorizer = CountVectorizer(ngram_range=(1,1),
                                          analyzer="word",
                                          #stop_words="english",
                                          lowercase=True,
                                          binary=True,
                                          max_features=500000)

        if tweet_test is None:
            feature = []
            for tweet in tweets:

                feature.append(tweet.tweet.source)


            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X = tfidfVectorizer.transform(feature)

            feature_names=tfidfVectorizer.get_feature_names()

            return X, feature_names
        else:
            feature  = []
            feature_test  = []
            for tweet in tweets:

                feature.append(tweet.tweet.source)

            for tweet in tweet_test:

                feature_test.append(tweet.tweet.source)


            tfidfVectorizer = tfidfVectorizer.fit(feature)

            X_train = tfidfVectorizer.transform(feature)
            X_test = tfidfVectorizer.transform(feature_test)

            feature_names=tfidfVectorizer.get_feature_names()

            return X_train, X_test, feature_names



##################################
#inizializer
def make_feature_manager():

    features_manager = Features_manager()

    return features_manager
