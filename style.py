import nltk
import string
from nltk.probability import FreqDist
from textblob import TextBlob
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show

class Text(object):
    def __init__(self, fn):
        # attributes live here
        self.filename = fn
        self.raw_text = self.get_raw_text()
        self.tokens = self.tokenize()
        self.processed_tokens = self.lowercase()
        self.processed_tokens = self.no_punctuation()
        self.processed_tokens = self.no_spaces()
        self.flattened_tokens = self.flatten(self.processed_tokens)
        self.stringified_text = self.get_stringified_text()
        self.stringified_sentences = self.get_stringified_sentences()
        self.sentiments = self.get_sentiment()
        self.sentiment_values = self.get_sentiment_values()

    # def bokeh_graph(self):
    #     output_file('bokeh_test.html')
    #     p = figure(plot_width=400, plot_height=400)
    #     p.line(self.sentiment_values, line_width=2)
    #     show(p)

    def graph_sentiment(self):
        plt.plot(self.sentiment_values)
        plt.ylabel('polarity')
        plt.xlabel('position in text')
        plt.title('Sentiment Analysis of Text')
        plt.show()

    def get_sentiment_values(self):
        return [val.polarity for val in self.sentiments]

    def get_sentiment(self):
        return [TextBlob(line).sentiment for line in self.stringified_sentences]


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
        new_tokens = [[item.lower() for item in each_token] for each_token in self.tokens]
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
        """Given the lists without any empty slots, turn the thing into one giant list of words and run a FreqDist on it"""
        new_text = []
        for line in self.processed_tokens:
            for item in line:
                new_text.append(item)
        fdist = nltk.FreqDist(new_text)
        modals = ['can', 'could', 'may', 'might', 'must', 'will']
        for m in modals:
            print(m + ':', fdist[m], end=' ')

def main():
    # get a filename
    # get raw text
    # preprocess that text:
    # 1. tokenize
    # 2. lowercase
    # 3. Get rid of punctuation
    # 4. get rid of white space
    # segment if you care (if you care about sentences,
    #   lines, stanzas, poems)
    # tag with parts of speech
    # do the stuff you actually care about
    filename = 'corpus/sabotage_clean.txt'
    our_text = Text(filename)
    our_text.raw_text
    # filename = 'corpus/sabotage_clean.txt'
    # raw_text = get_raw_text(filename)
    # tokens = tokenize(raw_text)
    # lower_tokens = lowercase(tokens)
    # without_punct = no_punctuation(lower_tokens)
    # without_spaces = no_spaces(without_punct)
    # fdist1 = frequency1(without_spaces)
    print(fdist1)



if __name__ == "__main__":
    main()
