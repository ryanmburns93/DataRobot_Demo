import fast_bert
from fast_bert.data_cls import BertDataBunch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flair.models import TextClassifier
from flair.data import Sentence
import string


def calculate_vader_sentiment(messages_df):
    # calculate score from vaderSentiment
    analyzer = SentimentIntensityAnalyzer()
    scored_messages_df = pd.DataFrame()
    for index, sentence_transcript_entry in enumerate(messages_df):
        vader_scores = analyzer.polarity_scores(sentence_transcript_entry['censoredShortBody'])
        temp_df = pd.DataFrame({'compound_polarity_VS': vader_scores['compound'],
                                'positive_ratio_VS': vader_scores['pos'],
                                'negative_ratio_VS': vader_scores['neg'],
                                'neutral_ratio_VS': vader_scores['neu']},
                               index=[0])
        temp_df = pd.concat([sentence_transcript_entry.to_frame().T, temp_df],
                            axis=1,
                            ignore_index=True)
        scored_messages_df = pd.concat([scored_messages_df, temp_df], axis=0, ignore_index=True)
    return scored_messages_df


def calculate_textblob_sentiment(messages_df):
    # calculate score from TextBlob - uses both the default PatternAnalyzer and the alternative NaiveBayesAnalyzer
    tb_sentiment_PA_polarity = [TextBlob(i).sentiment.polarity for i in messages_df['censoredShortBody']]
    tb_sentiment_PA_subj = [TextBlob(i).sentiment.subjectivity for i in messages_df['censoredShortBody']]
    tb_sentiment_NBA = [TextBlob(message, analyzer=NaiveBayesAnalyzer()).sentiment for message in messages_df['censoredShortBody']]
    tb_sentiment_NBA_class = [sentiment['classification'] for sentiment in tb_sentiment_NBA]
    tb_sentiment_NBA_pos_ratio = [sentiment['p_pos'] for sentiment in tb_sentiment_NBA]
    messages_df['sentiment_polarity_TB_PA'] = tb_sentiment_PA_polarity
    messages_df['sentiment_subjectivity_TB_PA'] = tb_sentiment_PA_subj
    messages_df['sentiment_classification_TB_NBA'] = tb_sentiment_NBA_class
    messages_df['sentiment_pos_percent_TB_NBA'] = tb_sentiment_NBA_pos_ratio
    return messages_df


def remove_stopwords_from_mesages(messages_df, filter_punctuation=True):
    stop_words = set(stopwords.words('english'))
    if filter_punctuation:
        stop_words = stop_words + list(string.punctuation)
    no_sw_message_list = []
    for index, sentence_transcript_entry in enumerate(messages_df):
        message_word_tokens = word_tokenize(sentence_transcript_entry['censoredShortBody'])
        filtered_sentence = [word for word in message_word_tokens if not word.lower() in stop_words]
        no_sw_message_list.append(filtered_sentence)
    messages_df['no_stopwords_message_text'] = no_sw_message_list
    return messages_df


def calculate_flair_sentiment(messages_df):
    classifier = TextClassifier.load('en-sentiment')
    flair_sentiment_score_list = []
    for message in messages_df['censoredShortBody']:
        message = Sentence(message)
        classifier.predict(message)
        value = message.labels[0].to_dict()['value']
        if value == 'POSITIVE':
            result = message.to_dict()['labels'][0]['confidence']
        else:
            result = -(message.to_dict()['labels'][0]['confidence'])
        result = round(result, 3)
        flair_sentiment_score_list.append(result)
    messages_df['flair_sentiment_score'] = flair_sentiment_score_list
    return messages_df

