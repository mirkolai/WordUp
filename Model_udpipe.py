# import es_core_news_md
import pandas as pd
from os.path import join
import ufal.udpipe
import conllu
import os
import spacy

class Model_udpipe:
    def __init__(self, path):
        """Load given model."""
        self.model = ufal.udpipe.Model.load(path)
        if not self.model:
            raise Exception("Cannot load UDPipe model from file '%s'" % path)

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

    def return_upostags(self,text):
        sentences = self.tokenize(text)

        # Then, we perform tagging and parsing for each sentence
        for s in sentences:
            self.tag(s)  # inplace tagging
            self.parse(s)  # inplace parsing

        conllu_txt = self.write(sentences, "conllu")  # conllu|horizontal|vertical

        # print(conllu_txt)
        output="FLAGupostags "
        conllu_obj = conllu.parse(conllu_txt)
        for i in range(0, len(sentences)):
            for item in conllu_obj[i]:
                if item['upostag'] not in ["_","X"]:
                    output+=" "+item['upostag']

        return output



    def return_deprelnegations(self, text):
        sentences = self.tokenize(text)

        # Then, we perform tagging and parsing for each sentence
        for s in sentences:
            self.tag(s)  # inplace tagging
            self.parse(s)  # inplace parsing

        conllu_txt = self.write(sentences, "conllu")  # conllu|horizontal|vertical

        # print(conllu_txt)
        output = "FLAGdeprelnegations "
        conllu_obj = conllu.parse(conllu_txt)
        for i in range(0, len(sentences)):
            for item in conllu_obj[i]:

                if item['feats'] is not None:
                    for key, value in item['feats'].items():
                        if value=="Neg":
                            output += " " + item['deprel']

        return output


    def return_deprels(self, text):
        sentences = self.tokenize(text)

        # Then, we perform tagging and parsing for each sentence
        for s in sentences:
            self.tag(s)  # inplace tagging
            self.parse(s)  # inplace parsing

        conllu_txt = self.write(sentences, "conllu")  # conllu|horizontal|vertical

        # print(conllu_txt)
        output = "FLAGdeprels "
        conllu_obj = conllu.parse(conllu_txt)
        for i in range(0, len(sentences)):
            for item in conllu_obj[i]:

                    output += " " + item['deprel']

        return output.replace(".","PUNCT").replace(":","")


    def return_relation(self, text,relation,key):
        sentences = self.tokenize(text)

        # Then, we perform tagging and parsing for each sentence
        for s in sentences:
            self.tag(s)  # inplace tagging
            self.parse(s)  # inplace parsing

        conllu_txt = self.write(sentences, "conllu")  # conllu|horizontal|vertical

        # print(conllu_txt)
        output = "FLAGrelationverbs"+key+relation+" "
        conllu_obj = conllu.parse(conllu_txt)
        for i in range(0, len(sentences)):
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


    def return_Sidorov(self, text,key):
        sentences = self.tokenize(text)

        # Then, we perform tagging and parsing for each sentence
        for s in sentences:
            self.tag(s)  # inplace tagging
            self.parse(s)  # inplace parsing

        conllu_txt = self.write(sentences, "conllu")  # conllu|horizontal|vertical

        # print(conllu_txt)
        output = "FLAGSidorov"+key+" "
        conllu_obj = conllu.parse(conllu_txt)
        for i in range(0, len(sentences)):
            for item in conllu_obj[i]:
                    if item["head"] is not None:
                        for item_loop in conllu_obj[i]:
                            if item_loop["id"]==item["head"]:
                                output += " "+item_loop[key]+item[key]

        return output.replace(".","PUNCT").replace(":","")




if __name__== "__main__":
    sentence_relations = []
    #model= Model_udpipe("udipe_model/spanish-ancora-ud-2.3-181115.udpipe")
    model = Model_udpipe("udipe_model/spanish-gsd-ud-2.3-181115.udpipe")
    #nlp = es_core_news_md.load()

    text = "gracias. una y mil veces por la programación de la TV nacional"
    sentences = model.tokenize(text)

    #Then, we perform tagging and parsing for each sentence
    for s in sentences:
        model.tag(s) #inplace tagging
        model.parse(s) #inplace parsing

    conllu_txt = model.write(sentences, "conllu")  # conllu|horizontal|vertical

    #print(conllu_txt)

    conllu_obj = conllu.parse(conllu_txt)
    for i in range(0,len(sentences)):
        print("sentence "+str(i))
        for item in conllu_obj[i]:
            print(item['deprel'])

            #for key in item.keys():
            #    print(key)


    #
