import nltk
import string
from nltk.probability import FreqDist

def get_raw_text(fn):
    """Given a filename, get the raw text"""
    with open(fn, 'r') as fin:
        raw_text = fin.readlines()
    return raw_text

def tokenize(line_text):
    """Given raw text that is a list of lines, produce the tokens"""
    line_tokens = []
    for line in line_text:
        line_tokens.append(nltk.word_tokenize(line))
    return line_tokens

def lowercase(all_tokens):
    """Given the tokenized text, lowercase everything"""
    new_tokens = [[item.lower() for item in each_token] for each_token in all_tokens]
    return new_tokens

def no_punctuation(text):
    """Given the lowercased text, remove punctuation"""
    new_text = [[''.join(c for c in s if c not in string.punctuation) for s in y] for y in text]
    return new_text

def no_spaces(text):
    """Given the text without punctuation, remove empty items from lists"""
    new_text = [[s for s in x if s] for x in text]
    return new_text

def frequency1(text):
    """Given the lists without any empty slots, turn the thing into one giant list of words and run a FreqDist on it"""
    new_text = []
    for line in text:
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

    filename = 'corpus/black_magic_clean.txt'
    raw_text = get_raw_text(filename)
    tokens = tokenize(raw_text)
    lower_tokens = lowercase(tokens)
    without_punct = no_punctuation(lower_tokens)
    without_spaces = no_spaces(without_punct)
    fdist1 = frequency1(without_spaces)
    print(fdist1)



if __name__ == "__main__":
    main()
