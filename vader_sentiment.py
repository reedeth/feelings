import nltk
import string
from textblob import TextBlob
import matplotlib.pyplot as plt
import os
import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# TODO: make your own classifier to be more nuanced
# TODO: train on our own data
# TODO: word2vec

class Text(object):
    def __init__(self, fn, the_raw_text=False):
        # attributes live here
        self.filename = fn
        if not the_raw_text:
            self.raw_text = self.get_raw_text()
        else:
            self.raw_text = the_raw_text.split('\n')
        self.tokens = self.tokenize()
        self.processed_tokens = self.lowercase()
        self.processed_tokens = self.no_punctuation()
        self.processed_tokens = self.no_spaces()
        self.flattened_tokens = self.flatten(self.processed_tokens)
        self.stringified_text = self.get_stringified_text()
        self.stringified_sentences = self.get_stringified_sentences()
        self.sentiments = self.get_sentiment(usetextblob=False)
        self.sentiment_values = self.get_sentiment_values()
        self.sentiments_with_lines = self.get_sentiment_with_lines()
        self.total_sentiment = self.get_total_sentiment()
        # self.lines_sorted_by_sentiment = self.get_lines_sorted_by_sentiment()
        # self.get_unsorted_csv_of_text()
        # self.most_positive = self.most_positive_five()
        # self.most_negative = self.most_negative_five()\
        # self.most_positive = self.sentiments_with_lines[-40:]
        # self.most_negative = self.sentiments_with_lines[:40]
        # self.graphed = self.graph_sentiment()

    def get_total_sentiment(self):
        """gives the average sentiment for a poem"""
        return sum(self.sentiment_values) / len(self.sentiment_values)

    def graph_sentiment(self):
        plt.plot(self.sentiment_values)
        plt.ylabel('polarity')
        plt.xlabel('position in text')
        plt.title('Sentiment Analysis of ' + self.filename)
        # plt.save('sentiment_graphs/' + self.filename + '.png')
        plt.show()

    def get_sentiment_with_lines(self, usetextblob=False):
        if usetextblob:
            sentiments = [(line, TextBlob(line).sentiment.polarity)
                          for line in self.stringified_sentences]
        else:
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

    def get_sentiment_values(self):
        # get all the compound values
        return [val['compound'] for val in self.sentiments]

    def get_sentiment(self, usetextblob=True):
        if usetextblob:
            return [TextBlob(line).sentiment
                for line in self.stringified_sentences]
        else:
            # example output for each sentence -
            # {'neg': 0.348, 'neu': 0.498, 'pos': 0.154, 'compound': -0.7096}
            analyzer = SentimentIntensityAnalyzer()
            return [analyzer.polarity_scores(line) for line in self.stringified_sentences]


    # methods live here
    def flatten(self, thing):
        return [item for sublist in thing for item in sublist]

    def get_stringified_sentences(self):
        return [' '.join(line) for line in self.processed_tokens]

    def get_stringified_text(self):
        return ' '.join(self.flattened_tokens)

    def get_raw_text(self):
        """Given a filename, get the raw text"""
        with open(self.filename, 'r') as fin:
            raw_text = fin.readlines()
        return raw_text

    def tokenize(self):
        """Given raw text that is a list of lines, produce the tokens"""
        line_tokens = []
        for line in self.raw_text:
            line_tokens.append(nltk.word_tokenize(line))
        return line_tokens

    def lowercase(self):
        """Given the tokenized text, lowercase everything"""
        new_tokens = [[item.lower()
                      for item in each_token] for each_token in self.tokens]
        return new_tokens

    def no_punctuation(self):
        """Given the lowercased text, remove punctuation"""
        new_text = [[''.join(c for c in s if c not in string.punctuation) for s in y] for y in self.processed_tokens]
        return new_text

    def no_spaces(self):
        """Given the text without punctuation, remove empty items from lists"""
        new_text = [[s for s in x if s] for x in self.processed_tokens]
        return new_text

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


class Corpus(object):
    def __init__(self, corpus_dir):
        self.dir = corpus_dir
        self.files = self.manifest()
        self.texts = self.make_texts()

    def manifest(self):
        """given a corpus directory, make indexed text objects from it"""
        texts = []
        for (root, _, files) in os.walk(self.dir):
            for fn in files:
                if fn[0] == '.':
                    pass
                else:
                    texts.append(os.path.join(root, fn))
        return texts

    def make_texts(self):
        the_texts = []
        for fn in self.files:
            with open(fn, 'r') as fin:
                raw_text = fin.read()
                # see if there are poems by looking for \n\n\n
                if '\n\n\n' in raw_text:
                    print('found a book')
                    the_poems = self.get_poems(fn)
                    text_objects_of_poems = [Text(fn, poem) for poem in the_poems]
                    the_texts.extend(text_objects_of_poems)
                else:
                    print('found a poem')
                    the_texts.append(Text(fn))
        return the_texts
        # return [Text(fn) for fn in self.files]

    def get_poems(self, text):
        """given a book file scoop out the poems"""
        with open(text, 'r') as fin:
            raw_text = fin.read()
            poems = raw_text.split('\n\n\n')
        return poems

# TODO: get metadata
# TODO: work with that metadata
# TODO: graph or cluster it


def main():
    corpus_dir = 'corpus/'
    the_corpus = Corpus(corpus_dir)
    print(len(the_corpus.texts))


# import vader_sentiment
# corpus = 'corpus/'
# the_corpus = vader_sentiment.Corpus(corpus)
# see how many poems we have
# len(the_corpus.texts)
# see the tokens for the sixth poem
# the_corpus.texts[5].tokens


if __name__ == "__main__":
    main()
