import pickle
from arm_tokenizer import tokenize
from nltk.tag.perceptron import PerceptronTagger

PICKLE = 'averaged_perceptron_tagger.pickle'

with  open('problem1.txt', encoding='utf-8') as file:
    text = file.read()
paragraphs = [content for content in text.strip().splitlines() if content]
num_of_paragr = (len(paragraphs))


our_tagger=PerceptronTagger(load=True)
print(our_tagger.tag(tokenize(text)))


