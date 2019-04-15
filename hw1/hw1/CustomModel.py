import math, collections
class CustomModel:

  def __init__(self, corpus):
    """Initial custom language model and structures needed by this mode"""
    self.UnigramCounts = collections.defaultdict(lambda: 0)
    self.BigramCounts = collections.defaultdict(lambda: 0)
    self.TrigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
    """  
    # TODO your code here
    for sentence in corpus.corpus:
        for i in range(1,len(sentence.data)):
            current = sentence.data[i].word
            if i>1:
                lastone = sentence.data[i-1].word
                self.BigramCounts[(lastone,current)] += 1
            if i>2:
                lastone = sentence.data[i-1].word
                lastsecond = sentence.data[i-2].word
                self.TrigramCounts[(lastsecond,lastone,current)] += 1
            self.UnigramCounts[current] += 1
            self.total += 1
                

  def score(self, sentence):
    """ With list of strings, return the log-probability of the sentence with language model. Use
        information generated from train.
    """
    # TODO your code here
    score = 0.0
    for i in range(3,len(sentence)):
        lastsecond = sentence[i-2]
        lastone = sentence[i-1]
        current = sentence[i]
        trigramcounts = self.TrigramCounts[(lastsecond,lastone,current)]
        bigramcounts = self.BigramCounts[(lastsecond,lastone)]
        unigramcounts = self.UnigramCounts[lastone]
        
        if trigramcounts>0:
            score += math.log(trigramcounts)
            score -= math.log(bigramcounts)
            continue
        else:
            bigramcounts = self.BigramCounts[(lastone,current)]
        
        if bigramcounts>0:
            score += math.log(bigramcounts)
            score -= math.log(unigramcounts)
            continue
        else:
            unigramcounts = self.UnigramCounts[current] + 1
        
        score += math.log(unigramcounts)
        score -= math.log(self.total + len(self.UnigramCounts))
    return score
