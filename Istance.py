import Tweet
#tweet_id,user_id,text,label

class Istance(object):

    tweet_id=None
    user_id=None
    text=None
    label=None
    tweet=None
    user=None

    def __init__(self, tweet_id, user_id,text,language,label,tweet,user):

        self.tweet_id=tweet_id
        self.user_id=user_id
        self.text = text
        self.language = language
        self.label=label
        self.tweet=tweet
        self.user=user

def make_istance(tweet_id, user_id,text,language,label,tweet,user):

    istance = Istance(tweet_id, user_id,text,language,label,tweet,user)

    return istance



