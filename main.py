import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

def train_model(company):
    twitter_data = pd.read_csv('twitter_training.csv', encoding='latin-1')
    tweets = twitter_data[['entity', 'tweet_text', 'sentiment']].dropna()
    tweets_of_interest = tweets[tweets['entity'] == company]
    print("Number of tweets about " + company + ": {}".format(len(tweets_of_interest)))
    print("Sample of data to be used (first 5): ")
    print(tweets_of_interest.head())
    train, test = train_test_split(tweets_of_interest, test_size=0.2)

    nb_pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])

    svm_pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SVC(kernel='linear'))
    ])

    lr_pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression())
    ])

    nb_pipeline.fit(train['tweet_text'], train['sentiment'])
    svm_pipeline.fit(train['tweet_text'], train['sentiment'])
    lr_pipeline.fit(train['tweet_text'], train['sentiment'])

    nb_pred = nb_pipeline.predict(test['tweet_text'])
    svm_pred = svm_pipeline.predict(test['tweet_text'])
    lr_pred = lr_pipeline.predict(test['tweet_text'])

    print("Other information about the model: \n")
    print("Naive Bayes classification report:")
    print(classification_report(test['sentiment'], nb_pred))
    print("Naive Bayes confusion matrix:")
    print(confusion_matrix(test['sentiment'], nb_pred))
    print()

    print("Support Vector Machine classification report:")
    print(classification_report(test['sentiment'], svm_pred))
    print("Support Vector Machine confusion matrix:")
    print(confusion_matrix(test['sentiment'], svm_pred))
    print()

    print("Logistic Regression classification report:")
    print(classification_report(test['sentiment'], lr_pred))
    print("Logistic Regression confusion matrix:")
    print(confusion_matrix(test['sentiment'], lr_pred))
    print()

    print("Naive Bayes accuracy score:{} %".format(accuracy_score(test['sentiment'], nb_pred) * 100))
    print("Support Vector Machine accuracy score: {} %".format(accuracy_score(test['sentiment'], svm_pred) * 100))
    print("Logistic Regression accuracy score: {} %".format(accuracy_score(test['sentiment'], lr_pred) * 100))

    return nb_pipeline, svm_pipeline, lr_pipeline

def predict_sentiment(tweet, nb_pipeline, svm_pipeline, lr_pipeline):
    print("Predicting the sentiment of the following tweet: '{}'".format(tweet))
    print("Naive Bayes prediction: {}".format(nb_pipeline.predict([tweet])))
    print("Support Vector Machine prediction: {}".format(svm_pipeline.predict([tweet])))
    print("Logistic Regression prediction: {} \n".format(lr_pipeline.predict([tweet])))


# Example usage:
company = 'Microsoft'
nb_pipeline, svm_pipeline, lr_pipeline = train_model(company)
print("\nTraining complete. You can now enter a tweet to predict its sentiment.")
input_tweet = [
    "Microsoft is the best company in the world. I love it.",
    "Microsoft is the worst company in the world. I hate it.",
    # you can even use some swear words to test the model
]

for tweet in input_tweet:
    predict_sentiment(tweet, nb_pipeline, svm_pipeline, lr_pipeline)