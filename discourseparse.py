import nltk
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict

class DiscourseParsing:
    def __init__(self):
        self.relations = {
            "conj_fol": {"but","however","nevertheless","otherwise","yet","still","nonetheless"},
            "conj_prev": {"till","until","despite","in spite", "though", "although"},
            "conj_infer": {"therefore","furthermore","consequently","thus","as a result","subsequently","eventually","hence"},
            "conditionals": {"if"},
            "strong_mod": {"might","could","can","would","may"},
            "weak_mod": {"should","ought to","need not","shall","will","must"},
            "neg": {"not","neither","never","no","nor"}
        }
        self.all_relations = self.relations["conj_fol"].union(self.relations["conj_prev"],self.relations["conj_infer"],self.relations["conditionals"],self.relations["strong_mod"],self.relations["weak_mod"],self.relations["neg"])
        self.tknzr = nltk.TweetTokenizer(reduce_len=True)
        self.stops = set(nltk.corpus.stopwords.words('english'))

    def discourse_parse(self,text):
        sent_text = nltk.sent_tokenize(text)
        sent_list = []
        for sent in sent_text:
            vectors = {
                "w": [],
                "f": [],
                "flip": [],
                "hyp": []
            }
            words = self.tknzr.tokenize(sent)
            for word in words:
                vectors["w"].append(word)
                vectors["f"].append(1)
                if word in self.relations["conditionals"] or word in self.relations["strong_mod"]:
                    hyp = 1
                else:
                    hyp = 0
                vectors["hyp"].append(hyp)
            for i, word in enumerate(words):
                vectors["flip"].append(1)
                if word in self.relations["conj_fol"] or word in self.relations["conj_infer"]:
                    n = 1
                    for k in words[i+1:]:
                        if k in self.all_relations:
                            break
                        else:
                            vectors["f"][i+n] += 1
                            n += 1
                elif word in self.relations["conj_prev"]:
                    for j,k in enumerate(words[:i]):
                        if k in self.all_relations:
                            break
                        else:
                            vectors["f"][j] += 1
                elif word in self.relations["neg"]:
                    for k in range(1,6):
                        try:
                            if words[i+k] in self.relations["conj_prev"] and words[i+k] in self.relations["conj_fol"]:
                                break
                            else:
                                vectors["flip"][i+k] -= 1
                        except IndexError:
                            break
            sent_list.append(vectors)
        vectors = {
            "w": [],
            "f": [],
            "flip": [],
            "hyp": []
        }
        for vects in sent_list:
            vectors["w"].extend(vects["w"])
            vectors["f"].extend(vects["f"])
            vectors["flip"].extend(vects["flip"])
            vectors["hyp"].extend(vects["hyp"])
        return vectors

    def vectorize(self,textlist:"""list of texts"""):
        words = set()
        texts = []
        for i,text in enumerate(textlist):
            texts.append(dict())
            lists = self.discourse_parse(text)
            for j, word in enumerate(lists["w"]):
                words.add(word)
                if word not in texts[i]:
                    texts[i][word] = {}
                    texts[i][word]["n"] = 1
                    texts[i][word]["f"] = lists["f"][j]
                    texts[i][word]["flip"] = lists['flip'][j]
                    texts[i][word]["hyp"] = lists["hyp"][j]
                else:
                    texts[i][word]["n"] += 1
                    texts[i][word]["f"] += lists["f"][j]
                    texts[i][word]['flip'] += lists['flip'][j]
                    texts[i][word]['hyp'] += lists['hyp'][j]
        words = words.difference(self.stops)
        the_index = {word: index for index,word in enumerate(list(words))}
        indptr = [0]
        indices = []
        data = []
        for d in texts:
            for term in d:
                if term in words:
                    index = the_index[term]*3
                    f = d[term]["f"]
                    flip = d[term]["flip"]/d[term]["n"]
                    hyp = d[term]["hyp"]/d[term]["n"]
                    data.append(f)
                    data.append(flip)
                    data.append(hyp)
                    indices.append(index)
                    indices.append(index+1)
                    indices.append(index+2)
            indptr.append(len(indices))
        return csr_matrix((data,indices,indptr),dtype=float)
