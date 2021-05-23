__author__ = 'mirko'
import numpy
import glob
import stopwordsiso as stopwords
import pickle
from gensim.models import FastText
import os
import re

from Database_manager import make_database_manager

class Linguistic_Words(object):

    pattern_split = re.compile(r"\W+")
    cue_words_categories={}
    categories=[]
    languages = { 'en' : "en_core_web_lg",'es' : "es_core_news_lg" }
    language=None
    embeddings=None

    def __init__(self,language):
        print("Loading linguistic words for "+language+"...")

        self.embeddings=FastText.load("resources/embeddings/"+language+"-embeddings")
        self.stop_words={}
        for word in stopwords.stopwords(language):
            self.stop_words[word]=0
        self.cue_words_categories = {}
        self.categories=[]
        self.language=language
        if os.path.isfile("cache/"+language+"_similar_words_cache.pickle"):
            infile = open("cache/"+language+"_similar_words_cache.pickle",'rb')
            self.similar_words_cache = pickle.load(infile)
            infile.close()
        else:
            self.similar_words_cache = {}
        file_names=glob.glob("resources/lessical/linguistic_words/*")
        #print(language,len(file_names))
        for file_name in file_names:
            concept=file_name.split("/")[-1].replace("_en_es_eu.csv","")
            self.categories.append(concept)
            self.cue_words_categories[concept]=[]
            file=open(file_name)
            for line in file:
                index=1 if language == "es" else 2
                word=line.replace("\n","").split(",")[index]
                if word in list(self.embeddings.wv.index_to_key):
                    self.cue_words_categories[concept].append(word)
        return

    def most_similar_words(self, word, topn=10):
        #print("spacy_most_similar",word)
        try:
            ms = self.embeddings.wv.most_similar(positive=[word],topn=topn)
        except:
            return ()
        return set([w[0].lower() for w in ms])

    def get_feature(self, tweet):

        values=[0]*len(self.cue_words_categories.keys())

        tokens = tweet.forms
        #print("tokens",tokens)
        for token in tokens:
            if token == "" or token in self.stop_words:
                continue
            #print("token",token)
            for category,words in self.cue_words_categories.items():
                if token in words:
                    values[self.categories.index(category)]+=1
                else:
                    for word in words:
                        if word not in self.similar_words_cache:
                            self.similar_words_cache[word]=self.most_similar_words(word, topn=10)
                            output=open("cache/"+self.language+"_similar_words_cache.pickle", "wb")
                            pickle.dump(self.similar_words_cache,output)
                            output.close()
                        #print(token, category, word, self.similar_words_cache[word])
                        if token in self.similar_words_cache[word]:
                            values[self.categories.index(category)]+=1
                            break
        return self.categories, values

if __name__ == '__main__':
    language="es"
    database_manager = make_database_manager(language,"train")
    instances = numpy.array(database_manager.return_istances())
    model = Linguistic_Words("es")

    for instance in instances:
        concepts, values=model.get_feature(instance)
        if numpy.sum(values)>0:
            print(instance.text)
            print(instance.forms)
            print(instance.lemmas)
            print("concepts", "values")
            print(concepts)
            print(values)
            print()
