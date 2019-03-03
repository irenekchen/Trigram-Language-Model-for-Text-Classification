import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2018
Homework 1 - Programming Component: Trigram Language Models
Daniel Bauer
"""

# Irene Chen ic2409

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """

    new_sequence = sequence
    new_sequence.append("STOP")
    if (n < 1 or n > len(sequence)):
        return []
    elif (n == 1):
        new_sequence.insert(0, "START")
    else:
        for i in range(n-2):
            new_sequence.insert(0, "START")
    #https://stackoverflow.com/questions/4112265/how-to-zip-lists-in-a-list
    ngrams = list(zip(*list(list(new_sequence[i:] for i in range(n))[j] for j in range(n))))
    for i in range(n-1):
        ngrams.pop(-1)
    return ngrams



class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 
        self.totalwordcount = 0

        for sentence in corpus:
            unigram = get_ngrams(sentence, 1)
            bigram = get_ngrams(sentence, 2)
            trigram = get_ngrams(sentence, 3)
            for item in unigram:
                self.totalwordcount += 1
                if item in self.unigramcounts.keys():
                    unigramcount = self.unigramcounts[item] + 1
                    self.unigramcounts[item] = unigramcount
                else:
                    self.unigramcounts[item] = 1
            self.totalwordcount -= 2
            for item in bigram:
                if item in self.bigramcounts.keys():
                    bigramcount = self.bigramcounts[item] + 1
                    self.bigramcounts[item] = bigramcount
                else:
                    self.bigramcounts[item] = 1
            if tuple(("START", "START")) in self.bigramcounts.keys():
                bigramcount = self.bigramcounts[tuple(("START", "START"))] + 1
                self.unigramcounts[tuple(("START", "START"))] = bigramcount
            else:
                self.bigramcounts[tuple(("START", "START"))] = 1
            for item in trigram:
                if item in self.trigramcounts.keys():
                    trigramcount = self.trigramcounts[item] + 1
                    self.trigramcounts[item] = trigramcount
                else:
                    self.trigramcounts[item] = 1            
        ##Your code here

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        if (trigram not in self.trigramcounts.keys()):
            return 0.0
        else:
            probability = self.trigramcounts[trigram] / self.bigramcounts[tuple((trigram[:-1]))]
        return probability

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if (bigram not in self.bigramcounts.keys()):
            return 0.0
        else:
            probability = self.bigramcounts[bigram] / self.unigramcounts[tuple((bigram[:-1]))]
        return probability
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        if (self.totalwordcount == 0 or unigram not in self.unigramcounts.keys()):
            return 0.0
        else:
            probability = self.unigramcounts[unigram] / self.totalwordcount
        return probability

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        probability = lambda1*self.raw_trigram_probability(trigram) + lambda2*self.raw_bigram_probability(tuple((trigram[1:]))) + lambda3*self.raw_unigram_probability(tuple((trigram[-1],)))
        return probability
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        sentence_num = 0
        sentence_probability = 0
        trigrams = get_ngrams(sentence, 3)
        for trigram in trigrams:
            sentence_probability += math.log2(self.smoothed_trigram_probability(trigram))
        return sentence_probability


    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        sum_logprob = 0
        totalwordcount = 0
        for sentence in corpus:
            for item in sentence:
                totalwordcount += 1
            sum_logprob += self.sentence_logprob(sentence)
        l = sum_logprob / 93002
        perplexity_score = 2 ** (-l)
        return perplexity_score


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       

        perplexity_score1 = []
        perplexity_score2 = []
        
        # high scoring essays
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if (pp1 < pp2):
                correct += 1
            total += 1
            # .. 


        # low scoring essays
        for f in os.listdir(testdir2):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            if (pp2 < pp1):
                correct += 1
            total += 1
            # .. 

        return correct/total

if __name__ == "__main__":

    get_ngrams(["natural", "language", "processing"], 1)
    model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)


    # Essay scoring experiment: 
    acc = essay_scoring_experiment('train_high.txt', 'train_low.txt', "test_high", "test_low")
    print(acc)

