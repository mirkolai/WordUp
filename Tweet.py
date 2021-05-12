
#tweet_id,user_id,retweet_count,favourite_count,source,created_at
class Tweet(object):

    tweet_id=None
    user_id=None
    retweet_count=None
    favourite_count=None
    source=None
    created_at=None

    def __init__(self,tweet_id,user_id,retweet_count,favourite_count,source,created_at):
            self.tweet_id=tweet_id
            self.user_id=user_id
            self.retweet_count=retweet_count
            self.favourite_count=favourite_count
            self.source=source
            self.created_at=created_at

def make_tweet(tweet_id,user_id,retweet_count,favourite_count,source,created_at):
    """
        Return a Tweet object.
    """
    tweet = Tweet(tweet_id,user_id,retweet_count,favourite_count,source,created_at)

    return tweet



