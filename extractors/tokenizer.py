from pycorenlp.corenlp import StanfordCoreNLP 


nlp = StanfordCoreNLP('http://10.5.18.109:9000')

def parse(text):
    #print(text)
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize,ssplit,pos,lemma',
        'outputFormat': 'json'
    })

    
    return output['sentences']

def dependency_parse(text):
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize,ssplit,pos,depparse,parse',
        'outputFormat': 'json'
    })
    return output


def ner(text):
    #print(text)
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize,ssplit,pos,lemma, ner',
        'outputFormat': 'json'
    })

    
    return output['sentences']



def openie(text):
    #print(text)
    output = nlp.annotate(text, properties={
        'annotators': 'tokenize,ssplit,pos,lemma, depparse, parse, natlog, openie',
        'outputFormat': 'json'
    })

    
    return output['sentences']







