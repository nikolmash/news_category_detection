import wget
import os

from joblib import load

TF_IDF_LINKS = ["https://www.dropbox.com/s/jepxzzyd982ef3n/tfidf_0_0.pkl?dl=1"]
MODEL_LINKS = ["https://www.dropbox.com/s/ev6ylo08fv0g70h/lr_0_0.joblib?dl=1"]


def download_model(stop_words, lemmatize, arr):
    index = stop_words + 2 * lemmatize
    wget.download(arr[index])


def get_model(stop_words, lemmatize):
    if not os.path.exists('./lr_%d_%d.joblib' % (stop_words, lemmatize)):
        download_model(stop_words, lemmatize, MODEL_LINKS)
    if not os.path.exists('./tfidf_%d_%d.pkl' % (stop_words, lemmatize)):
        download_model(stop_words, lemmatize, TF_IDF_LINKS)

    model = load('./lr_%d_%d.joblib' % (stop_words, lemmatize))
    tf_idf = load('./tfidf_%d_%d.pkl' % (stop_words, lemmatize))
    return tf_idf, model