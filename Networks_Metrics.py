class Networks_Metrics(object):

    user_id=None
    dimensions=None

    def __init__(self,user_id,dimensions):
            self.user_id=user_id
            self.dimensions=dimensions

def make_networks_metrics(user_id,dimensions):

    networks_metrics = Networks_Metrics(user_id,dimensions)

    return networks_metrics



