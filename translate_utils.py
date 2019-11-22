import unicodedata
import re
import tensorflow as tf

# convert unicode to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def preprocess(sentence):
    # convert to lowercase, strip any leading or trailing white space
    sentence = unicode_to_ascii(sentence.lower().strip()) 

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)

    sentence = sentence.strip()

    # add a start and end token to the sentence so that the model knows when to start and stop predicting
    sentence = "<start> " + sentence + " <end>"

    return sentence

# if given input in a single file with parallel translations, split the translation into two files
def separatePairs(input_path, output_path1, output_path2):
    input_lines = []
    output_lines1 = []
    output_lines2 = []

    with open(input_path, 'r', encoding = 'utf-8') as f:
        line = f.readline()

        while line:
            input_lines.append(line)
            line = f.readline()

    with open(output_path1, 'w', encoding = 'utf-8') as f_output1:
        with open(output_path2, 'w', encoding = 'utf-8') as f_output2:
            for lines in input_lines:
                lines = lines.strip().split("\t")
                f_output1.write(lines[0] + "\n")
                f_output2.write(lines[1] + "\n")

def create_dataset(path_file1, path_file2, num_sentences_to_use):
    eng_raw = []
    fra_raw = []

    eng_processed = []
    fra_processed = []

    with open(path_file1, 'r', encoding='utf-8') as f:
        eng_raw = f.readlines()

    with open(path_file2, 'r', encoding='utf-8') as f:
        fra_raw = f.readlines()

    for sentence in eng_raw[:num_sentences_to_use]:
        eng_processed.append(preprocess(sentence))

    for sentence in fra_raw[:num_sentences_to_use]:
        fra_processed.append(preprocess(sentence))

    return eng_processed, fra_processed

def tokenize(lang):
    word2int_mapper = tf.keras.preprocessing.text.Tokenizer(filters='')
    # creates dictionary that maps unique word to an int
    word2int_mapper.fit_on_texts(lang)   

    # takes each word and replaces it with corresponding int based on dictionary that was created
    word2int = lang_tokenizer.texts_to_sequences(lang)

    # pads shorter sentences with 0's to create uniformly sized sequences
    word2int = tf.keras.preprocessing.sequence.pad_sequences(word2int,padding='post')

    return word2int, word2int_mapper





