from nltk.tokenize import  RegexpTokenizer

HAPPY_EMOJI = ''':D+|:\)+|:-\]|:\]|:-3|:3|:->|:>|8-\)|8\)|:-\}|:\}|:o\)|:c\)|:\^\)|=\]|=\)|:-D+|8-D|8D|x-D|xD+|X-D|XD+|=D+|=3|B\^D|:-\)+|\^_\^'''
SAD_EMOJI = ''':‑\(|:\(+|:‑c|:c|:‑\<|:<|:‑\[|:\[|:-\|\||>:\[|:\{|:@|>:\('|:'‑\(|:'\('''
HAPPY_TEARS_EMOJI = ''':'-\)+|:'\)+'''
SAD_TEARS_EMOJI = """D‑:|D:\<|D:|D8|D;|D=|DX"""
SHOCK_EMOJI = ''':‑O+|:O+|:‑o+|:o+|:-0+|8‑0+|>:O+'''
KISS_EMOJI = ''':-\*|:\*|:×'''
SMIRK_EMOJI = ''';‑\)|;\)|\*-\)|\*\)|;‑\]|;\]|;\^\)|:‑,|;D"'''
TONGUE_EMOJI  = ''':‑P|:P|X‑P|XP|x‑p|xp|:‑p|:p|:‑Þ|:Þ|:‑þ|:þ|:‑b|:b|d:|=p|>:P'''
ANNOYED_EMOJI = r''':‑/|:/|:‑\.|>:\\|>:/|:\\|=/|=\\|:L|=L|:S'''
STRAINGHT_BLUSHING_COOL_EMOJI = ''':‑\||:\||:\$|\|;‑\)|\|‑O'''
SEALED_EMOJI = ''':‑X|:X|:‑#|:#|:‑&|:&'''
ANGEL_EMOJI = '''O:‑\)|O:\)|0:‑3|0:3|0:‑\)|0:\)|0;\^\)'''
EVIL_EMOJI = '''>:‑\)|>:\)|\}:‑\)|\}:\)|3:‑\)|3:\)|>;\)'''
OTHER_EMOJI = ''':‑J|#‑\)|%‑\)|%\)|:‑###\.\.|:###\.\.|<:‑\|'''
EMOJI= HAPPY_EMOJI + '|' + SAD_EMOJI + '|' + HAPPY_TEARS_EMOJI + '|' + SHOCK_EMOJI + '|' + KISS_EMOJI + '|' + SMIRK_EMOJI + '|' + TONGUE_EMOJI + '|' + ANNOYED_EMOJI + '|' + STRAINGHT_BLUSHING_COOL_EMOJI + '|' + OTHER_EMOJI + '|' + SEALED_EMOJI + '|' + ANGEL_EMOJI + '|' + EVIL_EMOJI + '|' + SAD_TEARS_EMOJI;
MULTIPLE_PUNCTUATION_MARKS1 = """(?<=[Ա-Ֆա-ֆև])\.{2,15}|(?<=[Ա-Ֆա-ֆև]):|(?<=[Ա-Ֆա-ֆև]),|(?<=[Ա-Ֆա-ֆև]);|:|(?<=[Ա-Ֆա-ֆև])…|\[|\{|\(|\}|\)|\]"""
ARM_PUNCTUATION_MARKS = "[Ա-Ֆա-ֆև]+՞[Ա-Ֆա-ֆև]+|[Ա-Ֆա-ֆև]+՚[Ա-Ֆա-ֆև]+|[Ա-Ֆա-ֆև]+՜[Ա-Ֆա-ֆև]+|[Ա-Ֆա-ֆև]+՛[Ա-Ֆա-ֆև]+|՟"
MULTIPLE_PUNCTUATION_MARKS2 = '''—|\.+|»|«|,|[\?!]+|…|-|։|⁚|∶|꞉|˸|:|\"|''|՛՛|“|”|‴|″|‹‹|››|„|❞|‟|˝|ՙ|՚"'''
NAMES_WITH_PUNCTUATION = '''[-\@#\$\.\.\.!\?\&\*\^ևԱ-Ֆա-ֆA-Za-z_:/\.0-9]{3,}'''
PARTICLES = '''«|[Ա-Ֆա-ֆև]+|»|(?<=['՛”‴″››„❞‟˝ՙ՚"»])-[Ա-Ֆա-ֆև]+'''
ABBREVIATED_NAMES = '''(?<= )[Ա-Ֆա-ֆև]+\.{0,1}-[Ա-Ֆա-ֆև]+|(?<= )[Ա-Ֆա-ֆև]+\.?'''
NUM_LETTER_EXPRESSIONS = '''[0-9\.\,]+-[Ա-Ֆա-ֆև]+'''
FLOATING_POINT_NUMBERS = '''[0-9]+\,[0-9]+|[0-9]+\.[0-9]+|[0-9]+/[0-9]+|[0-9]+:[0-9]+|[0-9]+'''
SPECIAL_CHARACTERS_WITH_NUMBERS = '''[0-9]+\,[0-9]+|[0-9]+\.[0-9]+|[0-9]+/[0-9]+|[0-9]+|€|%|\+|°С|[Ա-Ֆա-ֆև]+/[Ա-Ֆա-ֆև]+'''
FOREIGN_WORDS = "՞|՚|՜|՛|[։⁚∶꞉˸:։]|\S+";
TOKEN_PATTERN = EMOJI + '|' + MULTIPLE_PUNCTUATION_MARKS1 + '|' + ABBREVIATED_NAMES + '|' + NUM_LETTER_EXPRESSIONS + '|' + FLOATING_POINT_NUMBERS + '|' + SPECIAL_CHARACTERS_WITH_NUMBERS + '|' + ARM_PUNCTUATION_MARKS + '|' + PARTICLES + '|' + MULTIPLE_PUNCTUATION_MARKS2 + '|' + NAMES_WITH_PUNCTUATION +'|'+ FOREIGN_WORDS;
tokenizer = RegexpTokenizer(TOKEN_PATTERN)


def tokenize(text: str):
    return [(text[start:end]) for start, end in tokenizer.span_tokenize(text)]



