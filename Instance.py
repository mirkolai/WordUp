from Model_udpipe import Model_udpipe
#tweet_id,user_id,text,label
model_udpipe=None
class Instance(object):

    tweet_id=None
    user_id=None
    text=None
    label=None
    tweet=None
    user=None
    conllu_txt=None
    bio=None
    lessical_diversity=None
    networks_metrics=None
    def __init__(self, tweet_id, user_id,text,language,label,tweet,user,conllu_txt,bio, lessical_diversity,networks_metrics,networks_mds):
        model_udpipe=Model_udpipe.getInstance(language)
        self.tweet_id=tweet_id
        self.user_id=user_id
        self.text = text
        self.forms = model_udpipe.return_instances(conllu_txt,'form')
        self.lemmas = model_udpipe.return_instances(conllu_txt,'lemma')
        self.language = language
        self.label=label
        self.tweet=tweet
        self.user=user
        self.conllu_txt=conllu_txt
        self.bio=bio.bio
        self.screen_name=bio.screen_name

        self.upostag=model_udpipe.return_upostags(conllu_txt)
        self.deprelnegation=model_udpipe.return_deprelnegations(conllu_txt)
        self.deprels=model_udpipe.return_deprels(conllu_txt)
        self.relationVERB=model_udpipe.return_relation(conllu_txt,"VERB","form")
        self.relationNOUN=model_udpipe.return_relation(conllu_txt,"NOUN","form")
        self.relationADJ=model_udpipe.return_relation(conllu_txt,"ADJ","form")
        self.Sidorov_form=model_udpipe.return_Sidorov(conllu_txt,"form")
        self.Sidorov_upostag=model_udpipe.return_Sidorov(conllu_txt,"upostag")
        self.Sidorov_deprel=model_udpipe.return_Sidorov(conllu_txt,"deprel")

        self.target_context_one=model_udpipe.return_target_context_one(conllu_txt)
        self.target_context_two=model_udpipe.return_target_context_two(conllu_txt)



        self.lexical_diversity=lessical_diversity
        self.networks_metrics_base_centrality_friend=networks_metrics["base_friends_centrality"]
        self.networks_metrics_base_centrality_retweet=networks_metrics["base_retweets_centrality"]
        self.networks_metrics_augmented_centrality_retweet=networks_metrics["augmented_retweets_centrality"]
        self.networks_metrics_base_label_count_friend=networks_metrics["base_friends_label_count"]
        self.networks_metrics_base_label_count_retweet=networks_metrics["base_retweets_label_count"]
        self.networks_metrics_augmented_label_count_retweet=networks_metrics["augmented_retweets_label_count"]
        #self.networks_metrics_base_mds_retweet=networks_mds["base_retweets_mds"]


def make_istance(tweet_id, user_id,text,language,label,tweet,user,conllu_txt,bio,lessical_diversity,networks_metrics,networks_mds):

    instance = Instance(tweet_id, user_id, text, language, label, tweet, user,conllu_txt,bio,lessical_diversity,networks_metrics,networks_mds)

    return instance



