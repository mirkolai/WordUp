# import es_core_news_md
import numpy
import numpy as np
import pandas as pd
from os.path import join
import ufal.udpipe
import conllu
import os
import spacy
from gensim.models import FastText


class Model_udpipe:
    language=None
    model=None
    languages = { 'en' : "en_core_web_lg",'es' : "es_core_news_lg" }
    embeddings=None

    __instance = None
    @staticmethod
    def getInstance(language):
      """ Static access method. """
      if Model_udpipe.__instance == None:
         Model_udpipe(language)
      return Model_udpipe.__instance

    def __init__(self, language):
      """ Virtually private constructor. """
      if Model_udpipe.__instance != None:
         raise Exception("This class is a singleton!")
      else:
        Model_udpipe.__instance = self

        """Load given model."""
        if language not in ["es","eu"]:
            raise Exception("Language not supported")


        self.language=language

        path="resources/udpipe_models/"+self.language+"_udpipe_model.output"
        if self.model is None:
            print("loading ",path)
            self.model = ufal.udpipe.Model.load(path)
        if not self.model:
            raise Exception("Cannot load UDPipe model from file '%s'" % path)
        path="resources/embeddings/"+self.language+"-embeddings"
        if self.embeddings is None:
            print("loading ",path)
            self.embeddings=FastText.load("resources/embeddings/"+language+"-embeddings")
        if not self.embeddings:
            raise Exception("Cannot load FastText model from file '%s'" % path)

    def tokenize(self, text):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text, tokenizer)

    def read(self, text, in_format):
        """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
        input_format = ufal.udpipe.InputFormat.newInputFormat(in_format)
        if not input_format:
            raise Exception("Cannot create input format '%s'" % in_format)
        return self._read(text, input_format)

    def _read(self, text, input_format):
        input_format.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.model.tag(sentence,self.model.DEFAULT)

    def parse(self, sentence):
        """Parse the given ufal.udpipe.Sentence (inplace)."""
        self.model.parse(sentence, self.model.DEFAULT)

    def write(self, sentences, out_format):
        """Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""

        output_format = ufal.udpipe.OutputFormat.newOutputFormat(out_format)
        output = ''
        for sentence in sentences:
            output += output_format.writeSentence(sentence)
        output += output_format.finishDocument()
        return(output)

    def return_conllu_txt(self, text):
        sentences = self.tokenize(text)
        # Then, we perform tagging and parsing for each sentence
        for s in sentences:
            self.tag(s)  # inplace tagging
            self.parse(s)  # inplace parsing
        conllu_txt = self.write(sentences, "conllu")  # conllu|horizontal|vertical
        return conllu_txt

    def return_instances(self,conllu_txt,key):
        conllu_obj = conllu.parse(conllu_txt)
        output=[]
        for i in range(0, len(conllu_obj)):
            for item in conllu_obj[i]:
                if item[key]:
                    output.append(item[key])

        return output

    def return_upostags(self,conllu_txt):
        #Bag of upostag
        conllu_obj = conllu.parse(conllu_txt)
        output="FLAGupostags " #place holder per evitare feature vuota
        for i in range(0, len(conllu_obj)):
            for item in conllu_obj[i]:
                if item['upostag'] not in ["_","X"]:
                    output+=" "+item['upostag']

        return output

    def return_deprelnegations(self, conllu_txt):
        conllu_obj = conllu.parse(conllu_txt)

        output = "FLAGdeprelnegations "
        for i in range(0, len(conllu_obj)):
            for item in conllu_obj[i]:
                if item['feats'] is not None:
                    for key, value in item['feats'].items():
                        if value=="Neg":
                            output += " " + item['deprel']
        return output

    def return_deprels(self, conllu_txt):
        conllu_obj = conllu.parse(conllu_txt)
        output = "FLAGdeprels "
        for i in range(0, len(conllu_obj)):
            for item in conllu_obj[i]:
                    output += " " + item['deprel']
        return output.replace(".","PUNCT").replace(":","")

    def return_relation(self, conllu_txt,relation,key):
        conllu_obj = conllu.parse(conllu_txt)
        output = "FLAGrelationverbs"+key+relation+" "
        for i in range(0, len(conllu_obj)):
            for item in conllu_obj[i]:
                    if item["upostag"]==relation:

                        next_list=[]
                        for next in conllu_obj[i]:
                            if next["head"] is not None and item["id"]==next["head"]:
                                next_list.append(next)

                        previous_list=[]
                        for previous in conllu_obj[i]:
                            if item["head"] is not None and previous["id"]==item["head"]:
                                previous_list.append(previous)

                        if len(next_list) >0  and len(previous_list)>0:
                            for n in next_list:
                                for p in  previous_list:

                                    output += " "+p[key]+relation+n[key]
                        elif len(next_list) == 0  and len(previous_list)>0:
                            for p in previous_list:
                                output += " " + p[key] + relation
                        else:

                            for n in next_list:
                                output += " " + relation + n[key]

        return output.replace(".","PUNCT").replace(":","")

    def return_Sidorov(self, conllu_txt,key):
        conllu_obj = conllu.parse(conllu_txt)

        output = "FLAGSidorov"+key+" "
        for i in range(0, len(conllu_obj)):
            for item in conllu_obj[i]:
                    if item["head"] is not None:
                        for item_loop in conllu_obj[i]:
                            if item_loop["id"]==item["head"]:
                                output += " "+item_loop[key]+item[key]

        return output.replace(".","PUNCT").replace(":","")



    def return_target_context(self, conllu_txt):
        if self.language=="es":
            target="vaccin"
        else:
            target='txert'

        conllu_obj = conllu.parse(conllu_txt)
        previous_list = []
        next_list = []
        for i in range(0, len(conllu_obj)):
            for item in conllu_obj[i]:
                    if target in item["form"]:
                        for next in conllu_obj[i]:
                            if next["head"] is not None and item["id"]==next["head"]:
                                print("token",next['form'])
                                next_list.append(self.embeddings.wv.__getitem__(next["form"]))
                        for previous in conllu_obj[i]:
                            if item["head"] is not None and previous["id"]==item["head"]:
                               print("previous",previous['form'])
                               previous_list.append(self.embeddings.wv.__getitem__(previous["form"]))

                        print("next_list",type(next_list))
                        print("previous_list",type(previous_list))
        if len(next_list)==0:
            next_list.append(numpy.array([0]*self.embeddings.vector_size))
        if len(previous_list)==0:
            previous_list.append(numpy.array([0]*self.embeddings.vector_size))
        print("previous_list",np.average(previous_list,axis=0).shape)
        print("next_list",np.average(next_list,axis=0).shape)
        output = np.concatenate((np.average(previous_list,axis=0),
                                 np.average(next_list,axis=0)), axis=None)

        return output



    def return_target_context_two(self, conllu_txt):
        if self.language=="es":
            target="vaccin"
        else:
            target='txert'

        conllu_obj = conllu.parse(conllu_txt)
        previous_list = []
        previous_list_2 = []
        next_list = []
        next_list_2 = []
        for i in range(0, len(conllu_obj)):
            for item in conllu_obj[i]:
                    if target in item["form"]:
                        for next in conllu_obj[i]:
                            if next["head"] is not None and item["id"]==next["head"]:
                                print("token",next['form'])
                                next_list.append(self.embeddings.wv.__getitem__(next["form"]))
                                for next_2 in conllu_obj[i]:
                                    if next_2["head"] is not None and next["id"]==next_2["head"]:
                                        print("token",next_2['form'])
                                        next_list_2.append(self.embeddings.wv.__getitem__(next_2["form"]))


                        for previous in conllu_obj[i]:
                            if item["head"] is not None and previous["id"]==item["head"]:
                               print("previous",previous['form'])
                               previous_list.append(self.embeddings.wv.__getitem__(previous["form"]))
                               for previous_2 in conllu_obj[i]:
                                    if previous_2["head"] is not None and next["id"]==previous_2["head"]:
                                        print("token",previous_2['form'])
                                        next_list_2.append(self.embeddings.wv.__getitem__(previous_2["form"]))
                        print("next_list_2",type(next_list_2))
                        print("next_list",type(next_list))
                        print("previous_list_2",type(previous_list_2))
                        print("previous_list",type(previous_list))
        if len(next_list)==0:
            next_list.append(numpy.array([0]*self.embeddings.vector_size))
        if len(next_list_2)==0:
            next_list_2.append(numpy.array([0]*self.embeddings.vector_size))
        if len(previous_list)==0:
            previous_list.append(numpy.array([0]*self.embeddings.vector_size))
        if len(previous_list_2)==0:
            previous_list_2.append(numpy.array([0]*self.embeddings.vector_size))
        print("previous_list",np.average(previous_list,axis=0).shape)
        print("next_list",np.average(next_list,axis=0).shape)
        previous=np.concatenate(
                (np.average(previous_list,axis=0),
                 np.average(previous_list_2,axis=0)), axis=None)
        next=np.concatenate(
                (np.average(next_list,axis=0),
                 np.average(next_list_2,axis=0)), axis=None)
        output=np.concatenate(
                (previous,
                 next) )
        print("next.shape",next.shape)
        return output



if __name__== "__main__":

    #embeddings=FastText.load("resources/embeddings/es-embeddings")
    #print(embeddings.wv.most_similar(positive=['lorgfgxs'],topn=10))
    #p
    sentence_relations = []
    model = Model_udpipe("es")
    #nlp = es_core_news_md.load()

    text = "xxxxththf frgfa lorgfgxs vaccinos. una y mil veces por la programación de la TV nacional"
    conllu_txt=model.return_conllu_txt(text)
    print(conllu_txt)
    relation=''
    key=''
    output=model.return_target_context(conllu_txt)

    output=model.return_target_context_two(conllu_txt)

    print(output)
    print(type(output))
    print(output.shape)
    #print(len(conllu_obj),len(sentences))
    #for i in range(0,len(conllu_obj)):
    #    print("sentence "+str(i))
    #    for item in conllu_obj[i]:
    #        print(item)

