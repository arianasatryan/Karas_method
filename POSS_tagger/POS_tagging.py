from nltk.tag.perceptron import PerceptronTagger
import pyconll

PICKLE = 'averaged_perceptron_tagger.pickle'

def get_sents(data):
    tok_of_sent=[]
    sents=[]
    for sentence in data:
        for token in sentence:
            if token.upos is not None:
                tok_of_sent.append((token.form, token.upos))
        sents.append(tok_of_sent)
        tok_of_sent=[]
    return sents


UD_ARM_TRAIN = 'hy_armtdp-ud-train.conllu'
UD_ARM_TEST = 'hy_armtdp-ud-test.conllu'
train = pyconll.load_from_file(UD_ARM_TRAIN)
test = pyconll.load_from_file(UD_ARM_TEST)
train_sents = get_sents(train)
test_sents = get_sents(test)

def _get_pretrain_model():
    pos_tagger = PerceptronTagger(load=False)
    pos_tagger.train(sentences=train_sents,save_loc=PICKLE)
    print('Accuracy : ', pos_tagger.evaluate(test_sents))

_get_pretrain_model()