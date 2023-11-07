import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
import os


def load_df(input_file):
    ## Load the data
    df = pd.read_csv(input_file, sep=r'\|\*\|', on_bad_lines='warn', encoding='utf-8')
    # df = np.loadtxt(os.path.join(data_dir, input_file), delimiter="|*|", encoding='utf-8')
    # Check the shape of the dataframe. This will indicate the number of samples and the number of columns
    print("data loaded df.shape = ", df.shape)
    return df


def lemmatize_sent_tokens(words):
    pos_tags = nltk.pos_tag(words)
    # Lemmatize each word with appropriate POS tag
    lemmatized_words = []
    for word, pos in pos_tags:
        if pos.startswith('V'):
            lemmatized_words.append(lemmatizer.lemmatize(word, wordnet.VERB))
        elif pos.startswith('J'):
            lemmatized_words.append(lemmatizer.lemmatize(word, wordnet.ADJ))
        else:
            lemmatized_words.append(lemmatizer.lemmatize(word))

    return lemmatized_words


# cleans the data and writes it to a new file
def process(df):
    # filters out irrelevant comments by checking against relevant word bank
    check_list = ['vape', 'vaping', 'smoking', 'smoke', 'ecig', 'ecigarette', 'liquidnicotine', 'electronic', 'cigarette', 'e-juice', 'e-liquid', 'ejuice', 'eliquid', 'ehookah']
    df['comments'] = df['comments'].apply(lambda lst: [string for string in eval(lst) if any(word in string.lower() for word in check_list)])
    # df['comments'] = df['comments'].apply(lambda lst: [string for string in eval(lst)]) #  uncomment this line and comment out the two lines above if you do not want to filter out any data

    # replace urls, usernames, and subreddits with general terms
    df['string'] = df['comments'].apply(lambda x: [string for string in x])
    df['string'] = df['string'].apply(lambda x: [re.sub(r'http[s]?://[^\"\')\s]+', 'URL', s) for s in x])
    df['string'] = df['string'].apply(lambda x: [re.sub(r'[/]?u/[\w-]+', 'USERNAME', s) for s in x])
    df['string'] = df['string'].apply(lambda x: [re.sub(r'[/]?r/[\w-]+', 'SUBREDIT', s) for s in x])

    #removes special characters and numbers
    df['string'] = df['string'].apply(lambda x: [re.sub(r'\b\d+', '', s) for s in x])
    df['string'] = df['string'].apply(lambda x: [re.sub(r'[^\w\s.,\'_]', ' ', s) for s in x])

    #removes stopwords and lemmatizes
    stopwords = nltk.corpus.stopwords.words('english')
    df['tokens'] = df['string'].apply(lambda x: [word_tokenize(string) for string in x])
    df['tokens'] = df['tokens'].apply(lambda x: [[word for word in string if word not in stopwords] for string in x])
    df['tokens'] = df['tokens'].apply(lambda x: [lemmatize_sent_tokens(sent) for sent in x])
    df['processed_strings'] = df['tokens'].apply(lambda x: [' '.join(string) for string in x])

    #splits contractions then re processes
    df['processed_strings'] = df['processed_strings'].apply(lambda x: [re.sub(r'n\'t', 'not', s) for s in x])
    df['processed_strings'] = df['processed_strings'].apply(lambda x: [re.sub(r'\'s', 'is', s) for s in x])
    df['processed_strings'] = df['processed_strings'].apply(lambda x: [re.sub(r'\'m', 'am', s) for s in x])
    df['processed_strings'] = df['processed_strings'].apply(lambda x: [re.sub(r'\'d', 'had', s) for s in x])
    df['processed_strings'] = df['processed_strings'].apply(lambda x: [re.sub(r'\'re', 'are', s) for s in x])
    df['processed_strings'] = df['processed_strings'].apply(lambda x: [re.sub(r'\'ve', 'have', s) for s in x])
    df['processed_strings'] = df['processed_strings'].apply(lambda x: [re.sub(r'\'ll', 'will', s) for s in x])
    df['tokens'] = df['processed_strings'].apply(lambda x: [word_tokenize(string) for string in x])
    df['tokens'] = df['tokens'].apply(lambda x: [[word for word in string if word not in stopwords] for string in x])
    df['tokens'] = df['tokens'].apply(lambda x: [lemmatize_sent_tokens(sent) for sent in x])
    df['processed_strings'] = df['tokens'].apply(lambda x: [' '.join(string) for string in x])
    df['processed_strings'] = df['processed_strings'].apply(lambda x: [re.sub(r' ?\.', '', s) for s in x])
    df['processed_strings'] = df['processed_strings'].apply(lambda x: [re.sub(r' ?,', '', s) for s in x])

    # remove posts with 3 or less words
    df['processed_strings'] = df['processed_strings'].apply(lambda x: [comment for comment in x if len(comment.split(' ')) >= 3])
    print("Removed Short posts - " + str(df.shape))

    df.drop(columns=['ids'], inplace=True)
    df.drop(columns=['string'], inplace=True)
    df.drop(columns=['tokens'], inplace=True)


    ####### Output the processed data frame
    header = '|*|'.join(df.columns)
    print(header)
    np.savetxt(os.path.join(data_dir, "processed_" + str(input_file)), df, fmt=['%s', '%s', '%s'], delimiter="|*|", header=header, encoding='utf-8', comments='')
    print("Done")


if __name__ == "__main__":
    # initializes lemmatizer
    lemmatizer = WordNetLemmatizer()

    # specify file that data is in
    data_dir = "data"
    input_file = "reddit_data"
    data_file = os.path.join(data_dir, input_file)
    df = load_df(data_file)

    process(df)
