from nltk.tokenize import TweetTokenizer
import pandas as pd

class Emo:
    def __init__(self,
        lexicon_paths:dict,
        overfolder:str
    ):
        self.tknzr = TweetTokenizer()
        self.lexicons = {}
        for lex in lexicon_paths:
            filename = "../{}/{}".format(overfolder,lexicon_paths[lex])
            with open(filename,"r",encoding="utf-8") as f:
                if lex == "emotion":
                    self.lexicons[lex] = pd.read_csv(f,sep="\t",index_col=["Word","Sense"])
                elif lex == "affect intensity":
                    self.lexicons[lex] = pd.read_csv(f,sep="\t",index_col=["Word","AffectDimension"])
                else:
                    self.lexicons[lex] = pd.read_csv(f,sep="\t",index_col="Word")

    def classify_sentence(self,
        sentence:"""string or list of words""",
        database:"""one of VAD, emotion, or affect intensity""",
        tokenize:bool=True
    ):
        if database == "emotion":
            values = {
                "anger": 0,
                "anticipation": 0,
                "disgust": 0,
                "fear": 0,
                "joy": 0,
                "sadness": 0,
                "surprise": 0,
                "trust": 0
            }
        elif database == "VAD":
            values = {
                "Valence": 0,
                "Arousal": 0,
                "Dominance": 0
            }
        elif database == "affect intensity":
            values = {
                "anger": 0,
                "fear": 0,
                "joy": 0,
                "sadness": 0
            }
        else:
            raise ValueError("database must be one of VAD, emotion, or affect intensity")
        df = self.lexicons[database]
        words_factored = 0
        if tokenize:
            bag_of_words = self.tknzr.tokenize(sentence)
        else:
            if type(sentence) is str:
                bag_of_words = sentence.split()
            else:
                bag_of_words = sentence
        for word in bag_of_words:
            word = word.lower()
            try:
                word_values = df.loc[word]
            except KeyError:
                continue
            else:
                words_factored += 1
                for key in values:
                    if database == "VAD":
                        values[key] += df.loc[word,key]
                    elif database == "emotion":
                        values[key] += df.loc[(word,key),'Score']
                    elif database == "affect intensity":
                        try:
                            v = df.loc[(word,key),'Score']
                        except KeyError:
                            continue
                        else:
                            values[key] += v
        if words_factored == 0:
            #print("No words from this sentence found in lexicon.")
            return None
        return {key:(values[key]/words_factored) for key in values}, words_factored
    
    def classify(self,
        documents:"""list of documents (as strings or lists of words)""",
        database="all",
        tokenize:bool=True
    ):
        classification = {}
        if database in {"all","emotion"}:
            emo = True
            classification["emotion"] = {
                "anger": 0,
                "anticipation": 0,
                "disgust": 0,
                "fear": 0,
                "joy": 0,
                "sadness": 0,
                "surprise": 0,
                "trust": 0
            }
            emo_n = 0
        if database in {"all","VAD"}:
            vad = True
            classification["VAD"] = {
                "Valence": 0,
                "Arousal": 0,
                "Dominance": 0
            }
            vad_n = 0
        if database in {"all","affect intensity"}:
            aff = True
            classification["affect intensity"] = {
                "anger": 0,
                "fear": 0,
                "joy": 0,
                "sadness": 0
            }
            aff_n = 0
        for doc in documents:
            if emo:
                if (values := self.classify_sentence(doc,"emotion",tokenize)):
                    emo_n += 1
                    for k in values[0]:
                        classification["emotion"][k] += values[0][k]
            if vad:
                if (values := self.classify_sentence(doc,"VAD",tokenize)):
                    vad_n += 1
                    for k in values[0]:
                        classification["VAD"][k] += values[0][k]
            if aff:
                if (values := self.classify_sentence(doc,"affect intensity",tokenize)):
                    aff_n += 1
                    for k in values[0]:
                        classification["affect intensity"][k] += values[0][k]
        if emo:
            for key in classification["emotion"]:
                classification["emotion"][key] = classification["emotion"][key] / emo_n
        if vad:
            for key in classification["VAD"]:
                classification["VAD"][key] = classification["VAD"][key] / vad_n
        if aff:
            for key in classification["affect intensity"]:
                classification["affect intensity"][key] = classification["affect intensity"][key] / aff_n
        return classification