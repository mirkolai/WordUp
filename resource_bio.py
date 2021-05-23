class Bio(object):

    user_id=None
    bio=None
    screen_name=None

    def __init__(self, user_id, bio,screen_name):
            self.user_id=user_id
            self.bio=bio
            self.screen_name=screen_name

def make_bio(user_id, bio,screen_name):

    bio = Bio(user_id, bio,screen_name)

    return bio



