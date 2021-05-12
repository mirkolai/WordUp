import csv
from User import make_user
from Tweet import make_tweet
from Istance import make_istance
import glob
import os
import joblib

class Database_manager(object):

    language=None
    partition=None
    istances=None
    tweets=None
    users=None

    def __init__(self,language,partition):
        if language not in ["es","eu"]:
            exit("Language not supported")
        self.language=language
        if partition not in ["train","test"]:
            exit("partition not supported")
        self.partition=partition


    def return_istances(self):
        file_name="cache/" + self.language +'_' + self.partition + '.pkl'
        print("reading ",file_name)
        if self.istances is not None:
            pass
        elif os.path.isfile(file_name) :
            self.istances= joblib.load(file_name)
        else:
            self.istances=[]
            filelist = sorted(glob.glob("data/"+self.language+"_"+self.partition+"/"+self.language+"_"+self.partition+".csv"))
            for file in filelist:
                first = True
                csvfile=open(file, newline='')
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
                for istance in spamreader:
                    if not first:
                        tweet_id=istance[0]
                        user_id=istance[1]
                        text=istance[2]
                        label=istance[3]
                        tweet=self.return_tweet(tweet_id)
                        user=self.return_user(user_id)
                        this_istance=make_istance(tweet_id, user_id,text,self.language,label,tweet,user)

                        self.istances.append(this_istance)

                    first = False

            joblib.dump(self.istances, file_name)

        return self.istances

    def return_tweet(self, tweet_id):
        file_name="cache/" + self.language +'_tweet_' + self.partition + '.pkl'
        print("reading ",file_name,"for user ",tweet_id)
        if self.tweets is not None:
            pass
        elif os.path.isfile(file_name) :
            self.tweets= joblib.load(file_name)
        else:
            self.tweets=[]
            filelist = sorted(glob.glob("data/"+self.language+"_"+self.partition+"/"+self.language+"_tweet_"+self.partition+".csv"))
            for file in filelist:
                first = True
                csvfile=open(file, newline='')
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
                #tweet_id,user_id,retweet_count,favourite_count,source,created_at
                for tweet in spamreader:
                    if not first:
                            tweet_id=tweet[0]
                            user_id=tweet[1]
                            retweet_count=tweet[2]
                            favourite_count=tweet[3]
                            source=tweet[4]
                            created_at=tweet[5]

                            this_tweet=make_tweet(tweet_id,user_id,retweet_count,favourite_count,source,created_at)

                            self.tweets.append(this_tweet)

                    first = False

            joblib.dump(self.tweets, file_name)

        return next(filter(lambda x: (x.tweet_id == tweet_id), self.tweets),None)

    def return_user(self, user_id):
        file_name="cache/" + self.language +'_user_' + self.partition + '.pkl'
        print("reading ",file_name,"for user ",user_id)
        if self.users is not None:
            pass
        elif os.path.isfile(file_name) :
            self.users= joblib.load(file_name)
        else:
            self.users=[]
            filelist = sorted(glob.glob("data/"+self.language+"_"+self.partition+"/"+self.language+"_user_"+self.partition+".csv"))
            for file in filelist:
                first = True
                csvfile=open(file, newline='')
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
                #user_id,statuses_count,followers_count,friends_count,listed_count,created_at,emoji_in_bio
                for user in spamreader:
                    if not first:
                            user_id=user[0]
                            statuses_count=user[1]
                            followers_count=user[2]
                            friends_count=user[3]
                            listed_count=user[4]
                            created_at=user[5]
                            emoji_in_bio=user[6]

                            this_user=make_user(user_id,statuses_count,followers_count,friends_count,listed_count,created_at,emoji_in_bio)

                            self.users.append(this_user)

                    first = False

            joblib.dump(self.users, file_name)

        return next(filter(lambda x: (x.user_id == user_id), self.users),None)


def make_database_manager(language,partition):
    database_manager = Database_manager(language,partition)

    return database_manager




if __name__== "__main__":
    database_manager = Database_manager("es","train")
    istances=database_manager.return_istances()
    for istance in istances:
        print(istance.tweet.source)
        print(istance.user.followers_count)
