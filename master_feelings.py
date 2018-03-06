import nltk
import string
from textblob import TextBlob
import matplotlib.pyplot as plt
import os
import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob.classifiers import NaiveBayesClassifier
import sys
import argparse
import re

# TODO: train on our own data - shane is working on this.
# TODO: word2vec
# TODO: graph or cluster it
# TODO: first

# def feature_extractor(words):
# experimental - not implemented
#     """Default feature extractor for the NaiveBayesAnalyzer."""
#     return dict(((word, True) for word in words))
#
#
# def get_rotten_training_data():
# experimental - not implemented
#         neg_ids = nltk.corpus.movie_reviews.fileids('neg')
#         pos_ids = nltk.corpus.movie_reviews.fileids('pos')
#         neg_feats = [(feature_extractor(
#         nltk.corpus.movie_reviews.words(fileids=[f])), 'neg') for f in neg_ids]
#         pos_feats = [(feature_extractor(
#         nltk.corpus.movie_reviews.words(fileids=[f])), 'pos') for f in pos_ids]
#         train_data = neg_feats + pos_feats
#         return nltk.classify.NaiveBayesClassifier.train(train_data)


def train_classifier():
    with open('corpus/csvs/raw_training_set.csv', 'r') as fin:
        print('training')
        fin.read
        poemreader = csv.reader(fin, delimiter=',', quotechar='|')
        train = []
        for row in poemreader:
            try:
                if row[1] in ['pos', 'neg']:
                    train.append((row[0], row[1]))
            except IndexError:
                pass
        train = train[1:]

    cl = NaiveBayesClassifier(train)
    return cl, train

class Corpus(object):
    def __init__(self, corpus_dir, args):
        self.args = args
        self.dir = corpus_dir

        # get a list of all the filenames
        self.files = self.manifest()
        self.trained_classifier, self.training_data = self.train_classifier()
        # make texts from all the filenames
        self.texts = self.make_texts()
        self.authors = list(set([text.author for text in self.texts]))
        self.book_titles = list(set([text.book_title for text in self.texts]))
        self.average_sentiments = self.get_average_sentiment_with_lines()
        self.author_sentiments = self.get_author_averages()
        self.book_sentiments = self.get_book_averages()
        self.most_positive_poems_in_corpus = self.get_most_positive_poems_in_corpus()
        self.most_negative_poems_in_corpus = self.get_most_negative_poems_in_corpus()

# wants the top five pos and neg poems in an individual books
# get a poem based on first line

    def get_most_positive_poems_in_corpus(self):
        sorted_poems = self.average_sentiments
        sorted_poems.sort(key=lambda x: x[1])
        return sorted_poems[-4:]

    def get_most_negative_poems_in_corpus(self):
        sorted_poems = self.average_sentiments
        sorted_poems.sort(key=lambda x: x[1])
        return sorted_poems[:4]

    def get_average_sentiment_with_lines(self):
        return [(text.first_line, text.average_sentiment) for text in self.texts]

    def most_positive_poems_in_book(self, title_query):
        sub_corpus = [text for text in self.texts if text.book_title == title_query]
        sorted_poems = [(text.first_line, text.average_sentiment) for text in sub_corpus]
        sorted_poems.sort(key=lambda x: x[1])
        return sorted_poems[-4:]

    def most_negative_poems_in_book(self, title_query):
        sub_corpus = [text for text in self.texts if text.book_title == title_query]
        sorted_poems = [(text.first_line, text.average_sentiment) for text in sub_corpus]
        sorted_poems.sort(key=lambda x: x[1])
        return sorted_poems[:4]

    def get_poem_averages_given_book(self, title_query):
        return [(text.first_line, text.average_sentiment) for text in self.texts if text.book_title == title_query]

    def get_author_averages(self):
        results = {}
        for author in self.authors:
            results[author] = self.average_sentiment_for_subcorpus(self.author_query(author))
        return results

    def get_book_averages(self):
        results = {}
        for title in self.book_titles:
            results[title] = self.average_sentiment_for_subcorpus(self.title_query(title))
        return results

    def average_sentiment_for_subcorpus(self, sub_corpus):
        sentiment_readings = [text.average_sentiment for text in sub_corpus]
        return sum(sentiment_readings) / len(sentiment_readings)

    def author_query(self, query):
        return [text for text in self.texts if text.author == query]

    def title_query(self, query):
        return [text for text in
                self.texts if text.book_title == query]

    def first_line_query(self, query):
        for text in self.texts:
            if text.first_line == query:
                return text
        # return [text for text in
        #         self.texts if text.first_line == query]

    def train_classifier(self):
        with open('corpus/csvs/raw_training_set.csv', 'r') as fin:
            print('training')
            fin.read
            poemreader = csv.reader(fin, delimiter=',', quotechar='|')
            train = []
            for row in poemreader:
                try:
                    if row[1] in ['pos', 'neg']:
                        train.append((row[0], row[1]))
                except IndexError:
                    pass
            train = train[1:]

        cl = NaiveBayesClassifier(train)
        return cl, train

    def manifest(self):
        """given a corpus directory, make a list of filenames from it"""
        texts = []

        # for each tuple in this list of tuples (bc os.walk makes tuples),
        # rather than treating it as set of 3, assign new variable names
        # to each element of tuple. The "_" is a python convention that
        # lets you kind of throw that variable away. Passing files that
        # start with "." skips over weird hidden stuff in directories.
        # os.path.join crawls all our subdirectories to give us not just
        # names of files, but files with full directories attached (i.e.
        # "corpus/individual_poems/baraka_black_poems/etc")
        for (root, _, files) in os.walk(self.dir):
            for fn in files:
                if fn[0] == '.' or fn[-4:] == '.csv':
                    pass
                else:
                    texts.append(os.path.join(root, fn))
        return texts

    def make_texts(self):
        the_texts = []
        for fn in self.files:
            print(fn)
            with open(fn, 'r') as fin:

                # fin.read() reads every filename in self.files as one
                # giant string.
                raw_text = fin.read()

                # Looking at that big long string, see if there are
                # poems by looking for \n\n\n.
                print('found a book')
                the_poems = self.get_poems(raw_text)
                # For every poem in the_poems, turn it into a
                # Text object, then "extend" our list of giant poems
                # aka "the_texts" with all the new poems it finds.
                # We use extend and not append to make sure we're not
                # making a zillion lists inside our list.
                author, title = self.parse_filepath(fn)
                text_objects_of_poems = [Text(fn=fn, poem_text=poem, book_title=title, author=author, training_data=self.training_data, trained_classifier=self.trained_classifier, args=self.args) for poem in the_poems]
                the_texts.extend(text_objects_of_poems)
        return the_texts
        # return [Text(fn) for fn in self.files]

    def get_poems(self, text):
        """given a book file scoop out the poems"""
        # split takes a long string and returns a new list
        # seperated by whatever you say to split at.
        return text.split('\n\n\n')

    def get_raw_training_csv(self):
        if not os.path.exists('corpus/csvs'):
            os.makedirs('corpus/csvs')
        with open('corpus/csvs/raw_training_set.csv', 'w') as fout:
            csvwriter = csv.writer(fout)
            csvwriter.writerow(['TEXT', 'POLARITY'])
            # a corpus doesn't have stringified_sentences - so we'll need
            # to loop over every text to get the sentences
            for text in self.texts:
                for sent in text.stringified_sentences:
                    csvwriter.writerow([sent])

    def parse_filepath(self, filename):
        base = os.path.basename(filename)
        base_no_ext = os.path.splitext(base)[0]
        results = base_no_ext.split('-')
        author = results[0]
        title = results[1]
        return author, title

class Text(object):

        # The reason we have "the_raw_text=False" is to say that, when
        # making this Text object, the first item we pass it is its
        # filename, and IF it has a second item, that second item is
        # its is_a_poem_chunk. But the =False makes it so that if there is
        # no second item (if nothing is passed), we set the_raw_text equal
        # to False. Which matters bc we have it this if statement that
        # helps us distinguish between poems in poem-files and poems
        # in book-files. We're turning TWO different types of objects
        # BOTH into Text(objects).
    def __init__(self, fn, poem_text, book_title, author, training_data, trained_classifier, args):
        # attributes live here
        self.filename = fn
        self.book_title = book_title
        self.author = author
        self.training_data = training_data
        self.trained_classifier = trained_classifier
        self.raw_text = poem_text.split('\n')
        self.tokens = self.tokenize()
        self.processed_tokens = self.preprocess()
        self.flattened_tokens = self.flatten()
        self.stringified_text = self.get_stringified_text()
        self.stringified_sentences = self.get_stringified_sentences()
        self.first_line = self.stringified_sentences[0]
        self.sentiments = self.get_sentiment(usetextblob=args.usetextblob, usetrained=args.usetrainingdata, usevader=args.usevader)
        self.sentiment_values = self.get_sentiment_values(usetextblob=args.usetextblob, usetrained=args.usetrainingdata, usevader=args.usevader)
        self.sentiments_with_lines = self.get_sentiment_with_lines(usetextblob=args.usetextblob, usetrained=args.usetrainingdata, usevader=args.usevader)
        self.average_sentiment = self.get_average_sentiment()
        # self.lines_sorted_by_sentiment = self.get_lines_sorted_by_sentiment()
        # self.get_unsorted_csv_of_text()
        # self.most_positive = self.most_positive_five()
        # self.most_negative = self.most_negative_five()\
        # self.most_positive = self.sentiments_with_lines[-40:]
        # self.most_negative = self.sentiments_with_lines[:40]
        # self.graphed = self.graph_sentiment()

    def get_average_sentiment(self):
        """gives the average sentiment for a poem"""
        return sum(self.sentiment_values) / len(self.sentiment_values)

    def graph_sentiment(self):
        plt.plot(self.sentiment_values)
        plt.ylabel('polarity')
        plt.xlabel('position in text')
        plt.title('Sentiment Analysis of ' + self.filename)
        # plt.save('sentiment_graphs/' + self.filename + '.png')
        plt.show()

    def get_sentiment_with_lines(self, usetextblob, usetrained, usevader):
        if usetextblob == 'True':
            if usetrained == 'True':
                # Returns a list with two items: first item is a given line of
                # poem, second item is that line's sentiment score.
                results = []
                for line in self.stringified_sentences:
                    tb = TextBlob(line, classifier=self.trained_classifier)
                    results.append((line, tb.sentiment.polarity))
            else:
                # Returns a list with two items: first item is a given line of
                # poem, second item is that line's sentiment score.
                results = [(line, TextBlob(line).sentiment.polarity)
                              for line in self.stringified_sentences]
        elif usevader == 'True':
            # Returns a list of 5 items: line of poem; negative score;
            # neutral score; postitive score; compound score.
            analyzer = SentimentIntensityAnalyzer()
            results = []
            for line in self.stringified_sentences:
                line_scores = analyzer.polarity_scores(line)
                results.append((line, line_scores['neg'], line_scores['neu'], line_scores['pos'], line_scores['compound']))
        return results

    def get_lines_sorted_by_sentiment(self):
        pass
        # sorted_lines = self.sentiments_with_lines
        # sorted_lines.sort(key=lambda x: x[1])
        # return sorted_lines

    def get_unsorted_csv_of_text(self):
        if not os.path.exists('corpus/csvs'):
            os.makedirs('corpus/csvs')
        with open('corpus/csvs/black_art_csv.csv', 'w') as fout:
            csvwriter = csv.writer(fout)
            csvwriter.writerow(['TEXT', 'VALUE', 'Neg', 'Pos', 'Compound'])
            for text, neg, neu, pos, compound in self.sentiments_with_lines:
                csvwriter.writerow([text, neg, neu, pos, compound])

    def most_positive_five(self):
        # go over every line in the text
        pass
        # results = self.lines_sorted_by_sentiment[-40:]
        # for line, val in results:
        #     print(line)
        #     print(val)
        #     print('=====')
        # return results

    def most_negative_five(self):
        pass
        # results = self.lines_sorted_by_sentiment[:40]
        # for line, val in results:
        #     print(line)
        #     print(val)
        #     print('=====')
        # return results

    def get_sentiment_values(self, usetextblob, usetrained, usevader):
        if usetextblob == 'True':
            # self.sentiments is currently a list of TextBlob objects.
            # What this does is loop over every object and return a list
            # of float decimals corresponding to the polarity attribute?
            return [val.polarity for val in self.sentiments]
        elif usevader == 'True':
            # self.sentiments is currently a list of dictionaries. What
            # this does is to loop over every dictionary and return the
            # value for the 'compound' key, then make a list of all of
            # them.
            return [val['compound'] for val in self.sentiments]

    def get_sentiment(self, usetextblob, usetrained, usevader):
        if usetextblob == 'True':
            if usetrained == 'True':
                # Returns a list with two items: first item is a given line of
                # poem, second item is that line's sentiment score.
                results = []
                for line in self.stringified_sentences:
                    tb = TextBlob(line, classifier=self.trained_classifier)
                    results.append(tb.sentiment)
            # If we've set it above in the Text(obect) to use TextBlob,
            # this evaluates each line for sentiment. Example output
            # for each sentence is a list of TextBlob sentiment objects,
            # each of which has two? attributes (polarity and subjectivity)
            return results
        elif usevader == 'True':
            # If we've set it above in the Text(object) to use Vader,
            # this evaluates each line for sentiment. The example output
            # for each sentence is a dictionary with four entries:
            # {'neg': 0.348, 'neu': 0.498, 'pos': 0.154, 'compound':
            # -0.7096}
            analyzer = SentimentIntensityAnalyzer()
            return [analyzer.polarity_scores(line) for line in self.stringified_sentences]

    def flatten(self):
        # TextBlob is weird and wants long strings rather than
        # tokens. So here, we're basically undoing our tokenization
        # process.
        return [item for sublist in self.processed_tokens for item in sublist]

    def get_stringified_text(self):
        # Here we're taking our un-tokenized items and turning our item
        # into one GIANT string of words. This lets us analyze an entire
        # poem for one sentiment score.
        return ' '.join(self.flattened_tokens)

    def get_stringified_sentences(self):
        # Here we're taking our un-tokenized items and turning them into
        # a list of lines that match the lines in the poem. This lets us
        # analyze each line in a poem for sentiment scores.
        return [' '.join(line) for line in self.processed_tokens]

    def get_raw_text(self):
        """Given a filename, get the raw text"""
        with open(self.filename, 'r') as fin:
            raw_text = fin.readlines()
        return raw_text

    def tokenize(self):
        """Given raw text that is a list of lines, produce the tokens"""
        # If we want it to tokenize things differently, we have to make
        # our own version of "nltk.word_tokenize()" (i.e., if we want
        # it to care about punctuation or something)
        line_tokens = []
        for line in self.raw_text:
            line_tokens.append(nltk.word_tokenize(line))
        return line_tokens

    def preprocess(self):
        """take a list of tokenized lines and preprocess them by lowercasing, removing punctuation, and then throwing away empty elements"""
        processed_tokens = [[item.lower() for item in each_token] for
                            each_token in self.tokens]
        processed_tokens = [[''.join(c for c in s if c not in
                            string.punctuation) for s in y] for y in
                            processed_tokens]
        processed_tokens = [[s for s in x if s] for x in processed_tokens]
        return processed_tokens

    def frequency1(self):
        """Given the lists without any empty slots,
            turn the thing into one giant list of
            words and run a FreqDist on it"""
        new_text = []
        for line in self.processed_tokens:
            for item in line:
                new_text.append(item)
        fdist = nltk.FreqDist(new_text)
        modals = ['can', 'could', 'may', 'might', 'must', 'will']
        for m in modals:
            print(m + ':', fdist[m], end=' ')


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)

    # Formatted: 'abbreviation', 'actual arg', 'variable it's stored in'

    parser.add_argument('-tb', '--textblob', dest='usetextblob',
                        action='store',
                        help='Specify to use Textblob.', default='True')
    parser.add_argument('-td', '--trainingdata', dest='usetrainingdata',
                        action='store',
                    help='Specify to use training data of our own.', default='True')
    parser.add_argument('-v', '--vader', dest='usevader',
                        help='Specify True to use vader', default='False')

    return parser.parse_args(argv)

def main():
    args = parse_args()
    corpus_dir = 'corpus/'
    the_corpus = Corpus(corpus_dir, args)
    print(len(the_corpus.texts))

# how to run this in the interpreter
# import master_feelings
# args = master_feelings.parse_args()
# you can set the args in the interpreter like so (can also change the defaults in the parse_args function above)-
# args.usetrainingdata = 'True'
# corpus = 'corpus/'
# the_corpus = master_feelings.Corpus(corpus, args)
# see how many poems we have
# len(the_corpus.texts)
# see the tokens for the sixth poem
# the_corpus.texts[5].tokens
# this will tell you the average sentiment of each book
# the_corpus.book_sentiments
# using that, we can see what book we want to look at and give that name to another function:
# the_corpus.most_positive_poems_in_book('embryo')
# that gives us a list of each poem by average sentiment and with first line. you can use that first line to pull out a poem:
# text_we_care_about = the_corpus.first_line_query('come sing a song')
# text_we_care_about in this case is a text object, so we can do whatever text things we want, like get the sentiments with the lines.
# text_we_care_about.sentiments_with_lines

# to run from command line would be
# python3 master_feelings.py --usevader True
# or
# python3 master_feelings.py --usetextblob True --usetrainingdata True


if __name__ == "__main__":
    main()
