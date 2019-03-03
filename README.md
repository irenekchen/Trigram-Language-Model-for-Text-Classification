# Trigram-Language-Model-for-Text-Classification
An implementation of a trigram language model used for a text classification task.

# Extracting n-grams from a sentence
The function get_ngrams, takes a list of strings and an integer n as input, and returns padded n-grams over the list of strings. The result is a list of Python tuples. 

For example:
```
>>> get_ngrams(["natural","language","processing"],1)
[('START',), ('natural',), ('language',), ('processing',), ('STOP',)]
>>> get_ngrams(["natural","language","processing"],2)
('START', 'natural'), ('natural', 'language'), ('language', 'processing'), ('processing', 'STOP')]
>>> get_ngrams(["natural","language","processing"],3)
[('START', 'START', 'natural'), ('START', 'natural', 'language'), ('natural', 'language', 'processing'), ('language', 'processing', 'STOP')]
```

# Counting n-grams in a corpus
We use two different data sets for this task. The first data set is the Brown corpus, which is a sample of American written English collected in the 1950s. The format of the data is a plain text file brown_train.txt, containing one sentence per line. Each sentence has already been tokenized. brown_test.txt will be used to compute the perplexity of our language model. 

## Reading the Corpus and Dealing with Unseen Words 

The function corpus_reader in trigram_model.py takes the name of a text file as a parameter and returns a Python generator object. Generators allow you to iterate over a collection, one item at a time without ever having to represent the entire data set in a data structure (such as a list). This is a form of lazy evaluation. You could use this function as follows: 

```
>>> generator = corpus_reader("")
>>> for sentence in generator:
             print(sentence)

['the', 'fulton', 'county', 'grand', 'jury', 'said', 'friday', 'an', 'investigation', 'of', 'atlanta', "'s", 'recent', 'primary', 'election', 'produced', '``', 'no', 'evidence', "''", 'that', 'any', 'irregularities', 'took', 'place', '.']
['the', 'jury', 'further', 'said', 'in', 'term-end', 'presentments', 'that', 'the', 'city', 'executive', 'committee', ',', 'which', 'had', 'over-all', 'charge', 'of', 'the', 'election', ',', '``', 'deserves', 'the', 'praise', 'and', 'thanks', 'of', 'the', 'city', 'of', 'atlanta', "''", 'for', 'the', 'manner', 'in', 'which', 'the', 'election', 'was', 'conducted', '.']
['the', 'september-october', 'term', 'jury', 'had', 'been', 'charged', 'by', 'fulton', 'superior', 'court', 'judge', 'durwood', 'pye', 'to', 'investigate', 'reports', 'of', 'possible', '``', 'irregularities', "''", 'in', 'the', 'hard-fought', 'primary', 'which', 'was', 'won', 'by', 'mayor-nominate', 'ivan', 'allen', 'jr', '&', '.']
...
```

Note that iterating over this generator object works only once. After you are done, you need to create a new generator to do it again. 

There are two sources of data sparseness when working with language models: Completely unseen words and unseen contexts. One way to deal with unseen words is to use a pre-defined lexicon before we extract ngrams. The function corpus_reader has an optional parameter lexicon, which should be a Python set containing a list of tokens in the lexicon. All tokens that are not in the lexicon will be replaced with a special "UNK" token.

Instead of pre-defining a lexicon, we collect one from the training corpus. This is the purpose of the function get_lexicon(corpus). This function takes a corpus iterarator (as returned by corpus_reader) as a parameter and returns a set of all words that appear in the corpus more than once. The idea is that words that appear only once are so rare that they are a good stand-in for words that have not been seen at all in unseen text. 

When a new TrigramModel is created (in the __init__ method), we pass in the filename of a corpus file. We then iterate through the corpus twice: once to collect the lexicon, and once to count n-grams. 

## Counting n-grams

The method count_ngrams counts the occurrence frequencies for ngrams in the corpus. The method creates three instance variables of TrigramModel, which store the unigram, bigram, and trigram counts in the corpus. Each variable is a dictionary (a hash map) that maps the n-gram to its count in the corpus. 
For example, after populating these dictionaries, we want to be able to query

```
>>> model.trigramcounts[('START','START','the')]
5478
>>> model.bigramcounts[('START','the')]
5478
>>> model.unigramcounts[('the',)]
61428
```
Where model is an instance of TrigramModel that has been trained on a corpus. 

# Raw n-gram probabilities
raw_trigram_probability(trigram),  raw_bigram_probability(bigram), and 
raw_unigram_probability(unigram) each return an unsmoothed probability computed from the trigram, bigram, and unigram counts respectively. We keep track of the total number of words in order to compute the unigram probabilities. 

# Generating text
The method generate_sentence returns a list of strings, randomly generated from the raw trigram model. We keep track of the previous two tokens in the sequence, starting with ("START","START"). Then, to create the next word, we look at all words that appeared in this context and get the raw trigram probability for each.
A random word is drawn from this distribution and then added to the sequence. We stop generating words once the "STOP" token is generated. Here is a sample of some generated text: 

```
>>>model.generate_sentence()
['the', 'last', 'tread', ',', 'mama', 'did', 'mention', 'to', 'the', 'opposing', 'sector', 'of', 'our', 'natural', 'resources', '.', 'STOP']

>>> model.generate_sentence()
['the', 'specific', 'group', 'which', 'caused', 'this', 'to', 'fundamentals', 'and', 'each', 'berated', 'the', 'other', 'resident', '.', 'STOP']
```

The optional t parameter of the method specifies the maximum sequence length so that no more tokens are generated if the "STOP" token is not reached before t words. 

# Smoothed probabilities
The method smoothed_trigram_probability(self, trigram) uses linear interpolation between the raw trigram, unigram, and bigram probabilities. The defult interpolation parameters are set to lambda1 = lambda2 = lambda3 = 1/3. 

# Computing Sentence Probability
The method sentence_logprob(sentence), returns the log probability of an entire sequence. The function get_ngrams is used to compute trigrams and the smoothed_trigram_probability method is used to obtain probabilities. Each probability is converted into logspace using math.log2. For example: 

```
>>> math.log2(0.8)
-0.3219280948873623
```

Then we add the log probabilities. Regular probabilities would quickly become too small, leading to numeric issues, so we work with log probabilities instead. 

# Perplexity
The method perplexity(corpus), computes the perplexity of the model on an entire corpus. 
The perplexity is defined as 2-l, where l is defined as: 

![Perplexity Image](/perplexity.png)

Here M is the total number of words. So to compute the perplexity, sum the log probability for each sentence, and then divide by the total number of words in the corpus. 

If we run the perplexity function on the test set for the Brown corpus brown_test.txt, we get a perplexity less than 400. 

# Using the Model for Text Classification
We apply the trigram model to a text classification task. We use a data set of essays written by non-native speakers of English for the ETS TOEFL test. These essays are scored according to skill level low, medium, or high. We will only consider essays that have been scored as "high" or "low". We train a different language model on a training set of each category and then use these models to automatically score unseen essays. We compute the perplexity of each language model on each essay. The model with the lower perplexity determines the class of the essay. 
The files ets_toefl_data/train_high.txt and ets_toefl_data/train_low.txt in the data zip file contain the training data for high and low skill essays, respectively. The directories ets_toefl_data/test_high and ets_toefl_data/test_low contain test essays (one per file) of each category. 
The method essay_scoring_experiment is called by passing two training text files, and two testing directories (containing text files of individual essays) and returns the accuracy of the prediction. 
The method creates two trigram models, reads in the test essays from each directory, and computes the perplexity for each essay. It then compares the perplexities and the returns the accuracy (correct predictions / total predictions). 

On the essay data set, we se that we easily get an accuracy of > 80%.
