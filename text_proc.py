def clean_numbers(text):
    """Takes a text and gets rid of the line numbers"""
    processed_text = []
    for line in text:
        clean_line = []
        for char in line:
            if not char.isdigit():
                clean_line.append(char)
        processed_text.append(''.join(clean_line))
    return processed_text


def write_file(filename, text):
    """take the processed lines and recombine them in a new file"""
    with open(filename[:-4] + '_clean.txt', 'w') as fout:
        for line in text:
            fout.write(line)


def main():
    # get a file's text
    filename = 'corpus/sabotage.txt'
    # open the file and get the lines
    with open(filename, 'r') as fin:
        raw_text = fin.readlines()
    # take the raw lines and clean the numbers
    clean_text = clean_numbers(raw_text)
    # recombine them and write to a new file
    write_file(filename, clean_text)


if __name__ == "__main__":
    main()
