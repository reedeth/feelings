import nltk


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


def main():
    # get a filename
    # get raw text
    # preprocess that text:
    # 1. tokenize, lowercase, get rid of punctuation if you want.
    # 2. segment if you care (if you care about sentences,
    #   lines, stanzas, poems)
    # 3. tag with parts of speech
    # do the stuff you actually care about

    filename = 'corpus/sabotage_clean.txt'
    raw_text = get_raw_text(filename)
    tokens = tokenize(raw_text)
    print(tokens)


if __name__ == "__main__":
    main()
