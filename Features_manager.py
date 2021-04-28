from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
from scipy.sparse import csr_matrix, hstack



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
             "ngrams":          self.get_ngrams_features,
             #"chargrams":       self.get_nchargrams_features,
             # "puntuactionmarks":self.get_puntuaction_marks_features,
             # "capitalizedletters":self.get_capitalized_letters_features,
             # "laughter":self.get_laughter_features,
             # "upos":self.get_upos_features,
             # "deprelneg":self.get_deprelneg_features,
            # "deprel":self.get_deprel_features,
             # "relationformVERB"   :self.get_relationformVERB_features,
             # "relationformNOUN"   :self.get_relationformNOUN_features,
             # "relationformADJ"    :self.get_relationformADJ_features,
            #"Sidorovbigramsform"   :self.get_Sidorov_bigramsform_features,
            # "Sidorovbigramsupostag":self.get_Sidorov_bigramsupostag_features,
            #"Sidorovbigramsdeprel" :self.get_Sidorov_bigramsdeprel_features,

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



    def get_ngrams_features(self, tweets,tweet_test=None):
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
                len(re.findall(r"[A-Z][a-z]{1,}", tweet.text_accents_stripped)),
                len(re.findall(r"[A-Z]{2,}", tweet.text_accents_stripped)),
                len(re.findall(r"[a-z]{1,}[A-Z]{1,}[a-z]{1,}", tweet.text_accents_stripped)),]

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
                len(re.findall(r"[A-Z][a-z]{1,}", tweet.text_accents_stripped)),
                len(re.findall(r"[A-Z]{2,}", tweet.text_accents_stripped)),
                len(re.findall(r"[a-z]{1,}[A-Z]{1,}[a-z]{1,}", tweet.text_accents_stripped)),])


            for tweet in tweet_test:
                feature_test.append([
                len(re.findall(r"[A-Z][a-z]{1,}", tweet.text_accents_stripped)),
                len(re.findall(r"[A-Z]{2,}", tweet.text_accents_stripped)),
                len(re.findall(r"[a-z]{1,}[A-Z]{1,}[a-z]{1,}", tweet.text_accents_stripped)),])


            return csr_matrix(np.vstack(feature)),csr_matrix(np.vstack(feature_test)),\
                   ["feature_words_all_capital",
                    "feature_words_start_with_capital",
                    "feature_words_with_a_capital_letter_in_the_middle",]



    def get_laughter_features(self,tweets,tweet_test=None):


        if tweet_test is None:
            feature = []

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

#inizializer
def make_feature_manager():

    features_manager = Features_manager()

    return features_manager
