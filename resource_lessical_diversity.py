class Lessical_Diversity(object):

    tweet_id=None
    dimensions=None

    def __init__(self, tweet_id, dimensions):
            self.tweet_id=tweet_id
            self.dimensions=dimensions

def make_lessical_diversity(tweet_id, dimensions):

    lessical_diversity = Lessical_Diversity(tweet_id, dimensions)

    return lessical_diversity



