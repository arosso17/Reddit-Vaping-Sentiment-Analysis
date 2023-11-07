import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import os
import re


# plot the topics. Code taken directly from scikit-learn's demo:
# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
def plot_topics(model, feature_names, n_top_words, title):
    # Visualize the results
    # Plot top words
    fig, axes = plt.subplots(1, 3, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


# performs 3 topic modeling on a list of posts with the option to visualize using tsne or word clouds. Returns lists of comments separated by topic
def topic_modeling(list_of_lists_of_comments, tsne=False, word_cloud=False):
    # cleaned = df['processed_strings'].tolist()
    texts = list_of_lists_of_comments
    #vectorize the data
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()

    # Create an instance of LDA
    lda_model = LatentDirichletAllocation(n_components=3)

    # Fit the LDA model to your data
    lda_model.fit(X_tfidf)
    topic_word_distributions = lda_model.components_
    # print("topic_word_distributions = ", topic_word_distributions)
    document_topic_distributions = lda_model.transform(X_tfidf)
    # print("document_topic_distributions = ", document_topic_distributions)

    plot_topics(lda_model, vocab, 10, '3 Topics Found using LDA')
    topic_lists = [[] for _ in range(lda_model.n_components)]
    labels = []
    for i, doc_dist in enumerate(document_topic_distributions):
        topic = np.argmax(doc_dist)
        labels.append(topic)
        topic_lists[topic].append(texts[i])

    if tsne:
        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42, init='random', metric="cosine")
        X_tsne = tsne.fit_transform(X_tfidf)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels)
        plt.title("t-SNE Visualization of Clustering Results")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.show()

    if word_cloud:
        for t in topic_lists:
            cluster_text = ''.join(t)
            wc = WordCloud(width=1600, height=800, collocations=False, max_words=30).generate(cluster_text)
            default_colors = wc.to_array()
            plt.title("Custom colors")
            plt.imshow(wc.recolor(random_state=3))
            wc.to_file("wordcloud" + ".png")
            plt.show()
    return topic_lists


# performs sentiment analysis on a list of posts and returns 3 lists corresponding to negative posts, neutral posts, and positive posts
def sentiment_analysis(posts_lists, tsne=False):
    # Create an instance of the SentimentIntensityAnalyzer
    sentiment_analyzer = SentimentIntensityAnalyzer()

    # Determine the sentiment of the text.
    scores = [[], []]
    neg = []
    neu = []
    pos = []
    for comment in posts_lists:
        score = sentiment_analyzer.polarity_scores(comment)
        scores[0].append(score['compound'])
        scores[1].append(comment)
        if score['compound'] > 0.1:
            pos.append(comment)
        elif score['compound'] < -0.1:
            neg.append(comment)
        else:
            neu.append(comment)

    print("Average sentiment: ", np.mean(scores[0]))
    print("Number of positive: ", len(pos))
    print("Number of neutral: ", len(neu))
    print("Number of negative: ", len(neg))
    ## Most Positive
    # print(np.max(scores[0]))
    # print(scores[1][scores[0].index(np.max(scores[0]))])
    ## Most Negative
    # print(np.min(scores[0]))
    # print(scores[1][scores[0].index(np.min(scores[0]))])
    return neg, neu, pos


# returns the average sentiment of a list of comments
def sentiment_analysis_average(list_of_lists_of_comments):
    posts_lists = list_of_lists_of_comments
    if len(posts_lists) < 1:
        return 0
    sentiment_analyzer = SentimentIntensityAnalyzer()
    scores = []
    for comment in posts_lists:
        score = sentiment_analyzer.polarity_scores(comment)
        scores.append(score['compound'])
    return np.mean(scores)


# determines the sentiment of one string
def sentiment_analysis_single(text):
    sentiment_analyzer = SentimentIntensityAnalyzer()
    score = sentiment_analyzer.polarity_scores(text)
    return score['compound']


# removes the x most frequently used words from the corpus
def remove_top_x_words(list_of_lists, x):
    comments = []
    for list in list_of_lists:
        for text in eval(list):
            comments.append(text)
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(comments)
    vocab = vectorizer.get_feature_names_out()
    tfidf_array = X_tfidf.toarray()
    feature_sums = tfidf_array.sum(axis=0)
    top_x_indices = feature_sums.argsort()[:-(x+1):-1]
    print("Removing most popular words: ", vocab[top_x_indices])
    texts_without_top_words = []
    for text in comments:
        words = text.split()
        filtered_words = [word for word in words if word.lower().strip() not in vocab[top_x_indices]]
        filtered_text = " ".join(filtered_words)
        texts_without_top_words.append(filtered_text)
    return texts_without_top_words


def sentiment_then_cluster(comments, tsne=False):
    neg, neu, pos = sentiment_analysis(comments)
    print(len(neg))
    print(len(neu))
    print(len(pos))
    topic_modeling(neg, tsne)
    topic_modeling(neu, tsne)
    topic_modeling(pos, tsne)


def cluster_then_sentiment(comments):
    topics = topic_modeling(texts_without_top_words)
    for topic in topics:
        print(topic[0])
        sentiment_analysis(topic)


# determines if the sentiment of the discussion matches the sentiment of the post
def compare_sentiment_of_post_to_comments(df):
    ppcn = 0
    ppcm = 0
    ppcp = 0
    pmcn = 0
    pmcm = 0
    pmcp = 0
    pncn = 0
    pncm = 0
    pncp = 0
    totaln_sentiment = 0
    totalm_sentiment = 0
    totalp_sentiment = 0
    num = 0
    for index, row in df.iterrows():
        post_sentiment = sentiment_analysis_single(row['titles'])
        comments_sentiment = sentiment_analysis_average(eval(row["processed_strings"]))
        num += 1
        if post_sentiment > 0.1:
            if comments_sentiment > 0.1:
                ppcp += 1
                totalp_sentiment += comments_sentiment
            elif comments_sentiment > -0.1:
                ppcm += 1
                totalm_sentiment += comments_sentiment
            else:
                ppcn += 1
                totaln_sentiment += comments_sentiment
        elif post_sentiment > -0.1:
            if comments_sentiment > 0.1:
                pmcp += 1
                totalp_sentiment += comments_sentiment
            elif comments_sentiment > -0.1:
                pmcm += 1
                totalm_sentiment += comments_sentiment
            else:
                pmcn += 1
                totaln_sentiment += comments_sentiment
        else:
            if comments_sentiment > 0.1:
                pncp += 1
                totalp_sentiment += comments_sentiment
            elif comments_sentiment > -0.1:
                pncm += 1
                totalm_sentiment += comments_sentiment
            else:
                pncn += 1
                totaln_sentiment += comments_sentiment
    print("Average negative discussion sentiment: ", totaln_sentiment/num)
    print("Average neutral discussion sentiment: ", totalm_sentiment / num)
    print("Average positive discussion sentiment: ", totalp_sentiment / num)
    print('post positive comment positive: ', ppcp)
    print('post positive comment neutral: ', ppcm)
    print('post positive comment negative: ', ppcn)
    print('post neutral comment positive: ', pmcp)
    print('post neutral comment neutral: ', pmcm)
    print('post neutral comment positive: ', pmcn)
    print('post negative comment positive: ', pncp)
    print('post negative comment neutral: ', pncm)
    print('post negative comment negative: ', pncn)


# compares the number of comments on positive and negative sentiment posts
def check_engagement_of_pos_and_neg(df):
    number_of_pos_posts = 0
    number_of_neg_posts = 0
    num_of_coms_on_pos_posts = 0
    num_of_coms_on_neg_posts = 0
    for index, row in df.iterrows():
        post_sentiment = sentiment_analysis_single(row['titles'])
        if post_sentiment >= 0:
            number_of_pos_posts += 1
            num_of_coms_on_pos_posts += len(eval(row["processed_strings"]))
        else:
            number_of_neg_posts += 1
            num_of_coms_on_neg_posts += len(eval(row["processed_strings"]))

    print(number_of_pos_posts, num_of_coms_on_pos_posts)
    print(number_of_neg_posts, num_of_coms_on_neg_posts)
    print('positive engagement: ', num_of_coms_on_pos_posts / number_of_pos_posts)
    print('negative engagement: ', num_of_coms_on_neg_posts / number_of_neg_posts)


# loads the labeled data from the clustering analysis and runs a sentiment analysis on each cluster
def sentiment_per_cluster(file, tsne=False):
    df = pd.read_csv(file, sep=r'\|\*\|', on_bad_lines='warn', encoding='utf-8')
    dict_of_lables = {}
    for index, row in df.iterrows():
        if row["label"] in dict_of_lables.keys():
            dict_of_lables[row["label"]].append(row["comment"])
        else:
            dict_of_lables[row["label"]] = [row["comment"]]
    for label, comments in dict_of_lables.items():
        sentiment_analysis(comments, tsne=True)


if __name__ == "__main__":
    # Load our cleaned data
    data_dir = 'data'
    input_file = 'processed_reddit_data'
    df = pd.read_csv(os.path.join(data_dir, input_file), sep=r'\|\*\|', on_bad_lines='warn', encoding='utf-8')
    list_of_lists = df['processed_strings'].tolist()

    texts_without_top_words = remove_top_x_words(list_of_lists, 3)

    sentiment_then_cluster(texts_without_top_words, tsne=False)

    # sentiment_analysis(texts_without_top_words)

    # cluster_then_sentiment(texts_without_top_words)

    # compare_sentiment_of_post_to_comments(df)

    # check_engagement_of_pos_and_neg(df)

    # sentiment_per_cluster(os.path.join(data_dir, "labeled_" + input_file))
