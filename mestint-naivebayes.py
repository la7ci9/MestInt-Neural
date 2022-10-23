import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from matplotlib import pyplot as plt


def sentiment(value):
    if value == 1:
        return "positive"
    else:
        return "negative"

def cleanhtml(raw_html):
    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext


class NaiveBayesClassifier:
    def read_data(self):
        print('Running read_data...')
        # 1. Adatok beolvasasa
        self.df_train = pd.read_json("data/aclIMDB_train.json", orient="records")
        print()
        self.df_train.info()
        self.df_train.hist()
        print()
        print(self.df_train.head())

        self.df_test = pd.read_json("data/aclIMDB_test.json", orient="records")
        print()
        self.df_test.info()
        self.df_test.hist()
        print()
        print(self.df_test.head())
        # plt.show()

    def preprocessing(self):
        print('\nRunning preprocessing ...')
        # 2.1 sentiment oszlop készítése. Ahol a cimke 0 az oszlopban lévő mező értéke
        # legyen "negative" ellenkező esetben legyen "pozitive"
        self.df_train["sentiment"] = [None] * len(self.df_train)
        self.df_train["sentiment"] = self.df_train["label"].apply(sentiment)
        self.df_test["sentiment"] = [None] * len(self.df_test)
        self.df_test["sentiment"] = self.df_test["label"].apply(sentiment)
        print()
        print(self.df_test)

        # 2.2 Html tag-ek eltávolítása
        self.df_train["text"] = self.df_train["text"].apply(lambda x: cleanhtml(str(x)))

    def create_dict(self):
        print('\nRunning create_dict ...')
        # 3.1 Tokenizálás. (Szövegek szóközök mentén szavakra darabolása)
        nltk.download('stopwords')
        nltk.download('punkt')

        tokenized_reviews = self.df_train["text"].apply(lambda review_text: word_tokenize(review_text.lower()))

        print()
        print(tokenized_reviews.head())

        # 3.2 a tokenekből szótár építése a stopword-ök elhagyásával.
        d = dict()
        STOPWORDS = stopwords.words("english")

        for review in tokenized_reviews:
            for word in review:
                if word not in STOPWORDS and word.isalpha():
                    d[word] = d.get(word, 0) + 1

        # 3.3 szótár összerakása
        DESIRED_VOCAB_SIZE = 4000

        VOCAB = [k for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)[:DESIRED_VOCAB_SIZE]]
        self.word_table = pd.DataFrame({"word": VOCAB})
        print()
        print(self.word_table.head(10))

    def create_naive_bayes(self):
        print('\nRunning create_naive_bayes ...')
        # 4.1 Számoljuk meg a szótár elemeinek előfordulását és készítsünk egy
        # frekvenica táblázatott arról hogy egy szó mennyiszer fordul elő pozitv vagy negatív kontextusban
        VOCAB_IDX = {}
        dict_freqs = {"positive": {}, "negative": {}}

        # Hallgatókkal közösen
        for i in range(0, len(self.word_table["word"].values)):
            VOCAB_IDX[self.word_table["word"].values[i]] = i

        for idx in range(self.df_train.shape[0]):
            review = self.df_train.iloc[idx]["text"]
            sentiment = self.df_train.iloc[idx]["sentiment"]

            for word in review.split(" "):
                if word in VOCAB_IDX:
                    dict_freqs[sentiment][word] = dict_freqs[sentiment].get(word, 0) + 1

        # 4.2 Nézzük meg néhány szó előfrdulását
        # Hallgatókkal közösen
        # print()
        # print(dict_freqs["positive"]["good"])
        # print()
        # print(dict_freqs["negative"]["good"])
        # print()
        # print(dict_freqs["negative"]["bad"])
        # print()
        # print(dict_freqs["positive"]["bad"])

        # 4.3 Számoljuk ki az előfodulási gyakoriságok alapján az arányokat
        # total_positive = sum(dict_freqs["positive"].values())
        #
        # self.word_table["positive"] = [(dict_freqs["positive"].get(w, 0) + 1) / total_positive for w in self.word_table["word"]]
        #
        # total_negative = sum(dict_freqs["negative"].values())
        #
        # self.word_table["negative"] = [(dict_freqs["negative"].get(w, 0) + 1) / total_negative for w in self.word_table["word"]]
        #
        # print()
        # print(self.word_table.head())

        # 4.4 Szavak hangulati rátájának kiszámolása
        # self.word_table["ratio"] = np.log(self.word_table["positive"] / self.word_table["negative"])
        # print()
        # print(self.word_table.head())
        #
        # self.word_table = self.word_table.set_index("word")
        # print()
        # print(self.word_table.head())
        # print()
        # print(self.word_table["ratio"].describe())

    def apply_model(self):
        print("\nRunning apply_model ...")
        # 5.1 Előrejelzés egyszerű mondatokra

        def predict_for_review_raw(review):
            _input = cleanhtml(review)
            _input = word_tokenize(_input.lower())

            word_table_words = self.word_table.index

            return sum([self.word_table["ratio"].loc[token] for token in _input if token in word_table_words])

        print()
        print(predict_for_review_raw("This movie sucks."))

        print()
        print(predict_for_review_raw("This movie was fantastic!"))

        # 5.2 Model alkalmazása a tanuló adatokon
        def predict_for_review(review):
            return int(predict_for_review_raw(review) > 0)

        preds = self.df_train["text"].apply(predict_for_review)
        print()
        print(preds)

        real = (self.df_train["sentiment"] == "positive").astype(int)

        def get_accuracy(preds, real):
            return sum(preds == real) / len(real)

        print()
        print(f"Training set accuracy: {get_accuracy(preds, real)}")

        # 5.3 Model alkalmazása a test adatokon
        preds_test = self.df_test["text"].apply(predict_for_review)
        real_test = (self.df_test["sentiment"] == "positive").astype(int)
        print()
        print(f"Test set accuracy: {get_accuracy(preds_test, real_test)}")


def main():
    nbc = NaiveBayesClassifier()
    nbc.read_data()
    nbc.preprocessing()
    nbc.create_dict()
    nbc.create_naive_bayes()
    nbc.apply_model()


main()


