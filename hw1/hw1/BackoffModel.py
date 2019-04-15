import math, collections

class BackoffModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.UnigramCounts = collections.defaultdict(lambda: 0)
    self.BigramCounts = collections.defaultdict(lambda: 0)
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    # Tip: To get words from the corpus, try
    #    for sentence in corpus.corpus:
    #       for datum in sentence.data:  
    #         word = datum.word
    for sentence in corpus.corpus:
        for i in range(0,len(sentence.data)):
            current = sentence.data[i].word
            # Range from 0, so we need to make sure i>0 so that i-1 >= 0
            if i>0:
                lastone = sentence.data[i-1].word
                # Get Bigram count
                self.BigramCounts[(lastone,current)] += 1
            # Get Unigram count    
            self.UnigramCounts[current] += 1
            # Get N
            self.total += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0
    for i in range(1,len(sentence)):
        lastone = sentence[i-1]
        current = sentence[i]
        bigramcounts = self.BigramCounts[(lastone,current)]
        unigramcounts = self.UnigramCounts[lastone]
        
        if bigramcounts > 0:
            score += math.log(bigramcounts) # C(w_i|w_i-1)
            score -= math.log(unigramcounts) # C(w_i-1)
        else:
            unigramcounts = self.UnigramCounts[current] + 1 # Add 1 smoothing
            score += math.log(0.4*unigramcounts)
            score -= math.log(self.total + len(self.UnigramCounts)) # N+V
        
    return score
