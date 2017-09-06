import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


file1 = 'rt-polaritydata/rt-polarity.pos'
file2 = 'rt-polaritydata/rt-polarity.neg'
# Load data from files
data1 = list(open(file1, "rb").readlines())
data1 = [s.strip() for s in data1]
data1 = [clean_str(str(sent)) for sent in data1]

data2 = list(open(file2, "rb").readlines())
data2 = [s.strip() for s in data2]
data2 = [clean_str(str(sent)) for sent in data2]

data_with_label_pos = ['__label__1 ' + data1[i] for i in range(len(data1))]
data_with_label_neg = ['__label__0 ' + data2[i] for i in range(len(data2))]


number_train_pos = int(0.9 * len(data1))
with open('polarity_train.txt', 'w') as f1, open('polarity_valid.txt', 'w') as f2:
    for i in range(len(data1)):
        if i < number_train_pos:
            f1.write(data_with_label_pos[i] + '\n')
            f1.write(data_with_label_neg[i] + '\n')
        else:
            f2.write(data_with_label_pos[i] + '\n')
            f2.write(data_with_label_neg[i] + '\n')
