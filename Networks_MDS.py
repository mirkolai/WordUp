class Networks_MDS(object):

    user_id=None
    dimensions=None

    def __init__(self,user_id,dimensions):
            self.user_id=user_id
            self.dimensions=dimensions

def make_networks_mds(user_id,dimensions):

    networks_mds = Networks_MDS(user_id,dimensions)

    return networks_mds



