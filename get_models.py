import wget
import os

from joblib import load

TF_IDF_LINKS = ["https://www.dropbox.com/s/jepxzzyd982ef3n/tfidf_0_0.pkl?dl=1",
                "https://www.dropbox.com/s/q8rsynwc8arv3bv/tfidf_0_1.pkl?dl=1",
                "https://www.dropbox.com/s/73m7wqvi9q8gb1j/tfidf_1_0.pkl?dl=1",
                "https://www.dropbox.com/s/y6chuopdmp1ug8z/tfidf_1_1.pkl?dl=1"]
MODEL_LINKS = ["https://www.dropbox.com/s/ev6ylo08fv0g70h/lr_0_0.joblib?dl=1",
               "https://www.dropbox.com/s/5g0fbhqj4bsxedj/lr_0_1.joblib?dl=1"
               "https://www.dropbox.com/s/01wzyc7wtifyezl/lr_1_0.joblib?dl=1",
               "https://www.dropbox.com/s/zuxzlhrg1do46fb/lr_1_1.joblib?dl=1"]


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


def force_download():
    for stop_words in range(1):
        for lemmatize in range(1):
            if not os.path.exists('./lr_%d_%d.joblib' % (stop_words, lemmatize)):
                download_model(stop_words, lemmatize, MODEL_LINKS)
            if not os.path.exists('./tfidf_%d_%d.pkl' % (stop_words, lemmatize)):
                download_model(stop_words, lemmatize, TF_IDF_LINKS)


if __name__ == '__main__':
    force_download()
