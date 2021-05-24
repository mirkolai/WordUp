import csv
import time

from User import make_user
from Tweet import make_tweet
from Instance import make_istance
from Networks_Metrics import  make_networks_metrics
from Networks_MDS import  make_networks_mds
from Model_udpipe import Model_udpipe
import glob
import os
import joblib
import requests
import re
from resource_bio import make_bio
from resource_lexical_diversity import make_lexical_diversity


class Database_manager(object):

    language=None
    partition=None
    istances=None
    tweets=None
    model_udpipe=None
    users=None
    bios=None
    lexical_diversities=None
    networks_metrics=None
    networks_mds=None
    extended_url=None

    def __init__(self,language,partition):
        if language not in ["es","eu"]:
            raise Exception("Language not supported")
        self.language=language
        if partition not in ["train","test"]:
            raise Exception("partition not supported")
        self.partition=partition
        self.model_udpipe=Model_udpipe.getInstance(language)

    def return_istances(self):
        file_name="cache/" + self.language +'_' + self.partition + '.pkl'
        print("reading ",file_name)
        if self.istances is not None:
            pass
        elif os.path.isfile(file_name) :
            self.istances = joblib.load(file_name)
        else:
            self.istances=[]
            filelist = sorted(glob.glob("data/"+self.language+"_"+self.partition+"/"+self.language+"_"+self.partition+".csv"))
            for file in filelist:
                first = True
                csvfile=open(file, newline='')
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
                for instance in spamreader:
                    print(instance)
                    if not first:
                        if(len(instance)>0):
                            tweet_id=instance[0]
                            user_id=instance[1]
                            text=instance[2]
                            label=instance[3] if len(instance)==4 else None
                            tweet=self.return_tweet(tweet_id)
                            user=self.return_user(user_id)
                            conllu_txt=self.model_udpipe.return_conllu_txt(text)
                            bio=self.return_bio(user_id)

                            lexical_diversity=self.return_lexical_diversity(tweet_id)

                            networks_metrics={}
                            networks_metrics['base_friends_centrality']=self.return_networks_metrics(user_id,"base","friends","centrality")
                            networks_metrics['base_retweets_centrality']=self.return_networks_metrics(user_id,"base","retweets","centrality")
                            networks_metrics['augmented_retweets_centrality']=self.return_networks_metrics(user_id,"augmented","retweets","centrality")
                            networks_metrics['base_friends_label_count']=self.return_networks_metrics(user_id,"base","friends","label_count")
                            networks_metrics['base_retweets_label_count']=self.return_networks_metrics(user_id,"base","retweets","label_count")
                            networks_metrics['augmented_retweets_label_count']=self.return_networks_metrics(user_id,"augmented","retweets","label_count")
                            networks_mds={}
                            networks_mds['base_retweets_mds']=self.return_networks_mds(user_id,"base","retweets","mds")
                            networks_mds['base_friends_mds']=self.return_networks_mds(user_id,"base","friends","mds")
                            networks_mds['augmented_retweets_mds']=self.return_networks_mds(user_id,"augmented","retweets","mds")

                            this_istance=make_istance(tweet_id,
                                                      user_id,
                                                      text,
                                                      self.language,
                                                      label,
                                                      tweet,
                                                      user,
                                                      conllu_txt,
                                                      bio,
                                                      lexical_diversity,
                                                      networks_metrics,
                                                      networks_mds
                                                      )

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
            self.tweets= {}
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

                            self.tweets[tweet_id]=this_tweet

                    first = False

            joblib.dump(self.tweets, file_name)

        return self.tweets[tweet_id]

    def return_user(self, user_id):
        file_name="cache/" + self.language +'_user_' + self.partition + '.pkl'
        print("reading ",file_name,"for user ",user_id)
        if self.users is not None:
            pass
        elif os.path.isfile(file_name) :
            self.users= joblib.load(file_name)
        else:
            self.users= {}
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

                            self.users[user_id]=this_user

                    first = False

            joblib.dump(self.users, file_name)

        return self.users[user_id]

    def return_bio(self, user_id):
        file_name="cache/" + self.language +'_bio.pkl'
        print("reading ",file_name,"for user ",user_id)
        if self.bios is not None:
            pass
        elif os.path.isfile(file_name) :
            self.bios= joblib.load(file_name)
        else:
            self.bios= {}
            filelist = sorted(glob.glob("resources/bio/"+self.language+"_bio.csv"))
            for file in filelist:
                first = True
                csvfile=open(file, newline='')
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
                #user_id,statuses_count,followers_count,friends_count,listed_count,created_at,emoji_in_bio
                for user in spamreader:
                    if not first:
                            user_id=user[0]
                            screen_name=user[1]
                            bio=user[2]

                            this_user=make_bio(user_id, bio,screen_name)

                            self.bios[user_id]=this_user

                    first = False

            joblib.dump(self.bios, file_name)

        return  self.bios[user_id] if user_id in self.bios else make_bio(user_id, " NOBIOAVAILABLE "," NOSCREENNAMEAVAILABLE ")

    """#da finire"
    def return_extended_url(self,text):
        file_name="cache/" + self.language +'_extended_url_' + self.partition + '.pkl'
        print("reading ",file_name)
        urls=re.findall(r'(https?://\S+)', text)
        print("shorten urls",urls)
        if len(urls)==0:
            return None

        if self.extended_url is not None:
            pass
        elif os.path.isfile(file_name) :
            self.extended_url = joblib.load(file_name)
        else:
            self.extended_url={}
        if urls[0] in self.extended_url:
            return self.extended_url[urls[0]]
        else:
            try:
                session = requests.Session()  # so connections are recycled
                resp = session.head(urls[0], allow_redirects=True, timeout=7)
                full_url=resp.url
                print("full_url",full_url)
                if full_url==urls[0]:
                    p
                self.extended_url[urls[0]]=full_url
                time.sleep(3)
                joblib.dump(self.extended_url, file_name)
            except Exception as e:
                print(e)
                return None

        return self.extended_url[urls[0]]"""

    def return_lexical_diversity(self, tweet_id):

        file_name="cache/" + self.language +'_lexical_diversity_' + self.partition + '.pkl'
        print("reading ",file_name,"for tweet ",tweet_id)

        if os.path.isfile(file_name) :
            lexical_diversities= joblib.load(file_name)
        else:
            lexical_diversities= {}
            file = "resources/lexical/diversity/"+ self.language +'_lexical_diversity_' + self.partition + ".csv"
            first = True
            csvfile=open(file, newline='')
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            #user_id,statuses_count,followers_count,friends_count,listed_count,created_at,emoji_in_bio
            for tweet in spamreader:
                if first:
                    dimensions=tweet
                else:
                    current_user={}
                    tweet_id=tweet[0]
                    for i in range(2,len(dimensions)):
                        current_user[dimensions[i]]=tweet[i]

                    this_user=make_lexical_diversity(tweet_id,current_user)

                    lexical_diversities[tweet_id]=this_user

                first = False

        joblib.dump(lexical_diversities, file_name)

        return lexical_diversities[tweet_id]

    def return_networks_metrics(self, user_id, level, relation_type, measure_type):
        """level: base or augmented
           network_type: friends or retweets or retweets_timeline
           or label_count

           :return
           dimensions: name of dimension (header of csv file)
           values: values for the current user_id
        """
        file_name="cache/" + level +"_" + measure_type +"_" + self.language +'_' + relation_type + '_' + self.partition + '.pkl'
        print("reading ",file_name,"for user ",user_id)

        if os.path.isfile(file_name) :
            networks_metrics= joblib.load(file_name)
        else:
            networks_metrics= {}
            file = "resources/networks_metrics/" + level +"/" + measure_type +"_" + self.language +'_' + relation_type + '_' + self.partition + ".csv"
            first = True
            csvfile=open(file, newline='')
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            #user_id,statuses_count,followers_count,friends_count,listed_count,created_at,emoji_in_bio
            for user in spamreader:
                if first:
                    dimensions=user
                else:
                    current_user={}
                    user_id=user[0]
                    for i in range(1,len(dimensions)):
                        current_user[dimensions[i]]=user[i]

                    this_user=make_networks_metrics(user_id,current_user)

                    networks_metrics[user_id]=this_user

                first = False

        joblib.dump(networks_metrics, file_name)

        return networks_metrics[user_id]

    def return_networks_mds(self, user_id, level, relation_type, measure_type):
        """level: base or augmented
           network_type: retweets

           :return
           dimensions: name of dimension (header of csv file)
           values: values for the current user_id
        """
        file_name="cache/" + level +"_" + measure_type +"_" + self.language +'_' + relation_type + '_' + self.partition + '.pkl'
        print("reading ",file_name,"for user ",user_id)
        if os.path.isfile(file_name) :
            networks_mds= joblib.load(file_name)
        else:
            networks_mds= {}
            file = "resources/networks_mds/" + level +"/" + measure_type +"_" + self.language +'_' + relation_type + '_' + self.partition + ".csv"
            first = True
            csvfile=open(file, newline='')
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            #user_id,statuses_count,followers_count,friends_count,listed_count,created_at,emoji_in_bio
            for user in spamreader:
                if first:
                    dimensions=user
                else:
                    current_user={}
                    user_id=user[0]
                    for i in range(1,len(dimensions)):
                        current_user[dimensions[i]]=user[i]

                    this_user=make_networks_metrics(user_id,current_user)

                    networks_mds[user_id]=this_user

                first = False

        joblib.dump(networks_mds, file_name)

        return networks_mds[user_id]




def make_database_manager(language,partition):
    database_manager = Database_manager(language,partition)

    return database_manager




if __name__== "__main__":
    database_manager = Database_manager("eu","train")
    instances=database_manager.return_istances()
    for instance in instances:
        print(database_manager.return_extended_url(instance.text))
        #print(instance.networks_metrics_augmented_centrality_retweet.dimensions)
        #print(istance.user.followers_count)
        #print(istance.lessical_diversity)
        #print(istance.networks_metrics)
