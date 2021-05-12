
#user_id,statuses_count,followers_count,friends_count,listed_count,created_at,emoji_in_bio

class User(object):

    user_id=None
    statuses_count=None
    followers_count=None
    friends_count=None
    listed_count=None
    created_at=None
    emoji_in_bio=None


    def __init__(self,user_id,statuses_count,followers_count,friends_count,listed_count,created_at,emoji_in_bio):
        self.user_id=user_id
        self.statuses_count=statuses_count
        self.followers_count=followers_count
        self.friends_count=friends_count
        self.listed_count=listed_count
        self.created_at=created_at
        self.emoji_in_bio=emoji_in_bio

def make_user(user_id,statuses_count,followers_count,friends_count,listed_count,created_at,emoji_in_bio):
    """
        Return a Tweet object.
    """
    user = User(user_id,statuses_count,followers_count,friends_count,listed_count,created_at,emoji_in_bio)

    return user



