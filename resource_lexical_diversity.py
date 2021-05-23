class Lexical_Diversity(object):

    tweet_id=None
    dimensions=None

    def __init__(self, tweet_id, dimensions):
            self.tweet_id=tweet_id
            self.dimensions=dimensions

def make_lexical_diversity(tweet_id, dimensions):

    lexical_diversity = Lexical_Diversity(tweet_id, dimensions)

    return lexical_diversity



