import csv
from Tweet import make_tweet
import glob
import os
import joblib

class Database_manager(object):

    language=None

    def __init__(self,language):
        if language not in ["es","eu"]:
            exit("Language not supported")
        self.language=language

    def return_tweets(self):
        tweets=self.return_tweets_training()+self.return_tweets_test()
        return tweets

    def return_train(self):

        if os.path.isfile("cache/"+self.language+'_train.pkl') :
            tweets= joblib.load("cache/"+self.language+'_train.pkl')
            return tweets
        tweets=[]
        filelist = sorted(glob.glob("data/"+self.language+"_train/"+self.language+"_train.csv"))
        for file in filelist:
            first = True
            csvfile=open(file, newline='')
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for tweet in spamreader:
                if not first:
                        tweet_id=tweet[0]
                        user_id=tweet[1]
                        text=tweet[2]
                        language=self.language
                        label=tweet[3]
                        """
                        Create a new istance of a Tweet object
                        #tweet_id,user_id,text,label
                        """
                        this_tweet=make_tweet(tweet_id, user_id, text, language, label)

                        tweets.append(this_tweet)

                first = False

        joblib.dump(tweets, "cache/"+self.language+'_train.pkl')

        return tweets


    def return_tweet_train(self):

        if os.path.isfile("cache/"+self.language+'_tweet_train.pkl') :
            tweets= joblib.load("cache/"+self.language+'_tweet_train.pkl')
            return tweets
        tweets=[]
        filelist = sorted(glob.glob("data/"+self.language+"_train/"+self.language+"_tweet_train.csv"))
        for file in filelist:
            first = True
            csvfile=open(file, newline='')
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for tweet in spamreader:
                if not first:
                        tweet_id=tweet[0]
                        user_id=tweet[1]
                        text=tweet[2]
                        language=self.language
                        label=tweet[3]
                        """
                        Create a new istance of a Tweet object
                        #tweet_id,user_id,text,label
                        """
                        this_tweet=make_tweet(tweet_id, user_id, text, language, label)

                        tweets.append(this_tweet)

                first = False

        joblib.dump(tweets, "cache/"+self.language+'_tweet_train.pkl')

        return tweets


#da fare il test, uguale al train



def make_database_manager(language):
    database_manager = Database_manager(language)

    return database_manager




if __name__== "__main__":
    database_manager = Database_manager("en")

    tweets=database_manager.return_tweets_training()
    print("Tweets train")
    for tweet in tweets:
        print(tweet.id,tweet.text,tweet.language,tweet.label,tweet.topic)
    tweets=database_manager.return_tweets_test()
    print("Tweets test")
    for tweet in tweets:
        print(tweet.id, tweet.text, tweet.language, tweet.label, tweet.topic)

    tweets = database_manager.return_tweets()
    print("Tweets")
    for tweet in tweets:
        print(tweet.id, tweet.text, tweet.language, tweet.label, tweet.topic)
