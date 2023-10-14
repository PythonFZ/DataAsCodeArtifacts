import os
import pickle
import sys

import numpy as np
import package1
import pandas as pd
import scipy.sparse as sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def get_df(data):
    """Read the input data file and return a data frame."""
    df = pd.read_csv(
        data,
        encoding="utf-8",
        header=None,
        delimiter="\t",
        names=["id", "label", "text"],
    )
    sys.stderr.write(f"The input data frame {data} size is {df.shape}\n")
    return df


def save_matrix(df, matrix, names, output):
    """
    Save the matrix to a pickle file.

    Args:
        df (pandas.DataFrame): Input data frame.
        matrix (scipy.sparse.csr_matrix): Input matrix.
        names (list): List of feature names.
        output (str): Output file name.
    """
    id_matrix = sparse.csr_matrix(df.id.astype(np.int64)).T
    label_matrix = sparse.csr_matrix(df.label.astype(np.int64)).T

    result = sparse.hstack([id_matrix, label_matrix, matrix], format="csr")

    msg = "The output matrix {} size is {} and data type is {}\n"
    sys.stderr.write(msg.format(output, result.shape, result.dtype))

    with open(output, "wb") as fd:
        pickle.dump((result, names), fd)
    pass


def generate_and_save_train_features(train_input, train_output, bag_of_words, tfidf):
    """
    Generate train feature matrix.

    Args:
        train_input (str): Train input file name.
        train_output (str): Train output file name.
        bag_of_words (sklearn.feature_extraction.text.CountVectorizer): Bag of words.
        tfidf (sklearn.feature_extraction.text.TfidfTransformer): TF-IDF transformer.
    """
    df_train = get_df(train_input)
    train_words = np.array(df_train.text.str.lower().values)

    bag_of_words.fit(train_words)

    train_words_binary_matrix = bag_of_words.transform(train_words)
    feature_names = bag_of_words.get_feature_names_out()

    tfidf.fit(train_words_binary_matrix)
    train_words_tfidf_matrix = tfidf.transform(train_words_binary_matrix)

    save_matrix(df_train, train_words_tfidf_matrix, feature_names, train_output)


def generate_and_save_test_features(test_input, test_output, bag_of_words, tfidf):
    """
    Generate test feature matrix.

    Args:
        test_input (str): Test input file name.
        test_output (str): Test output file name.
        bag_of_words (sklearn.feature_extraction.text.CountVectorizer): Bag of words.
        tfidf (sklearn.feature_extraction.text.TfidfTransformer): TF-IDF transformer.
    """
    df_test = get_df(test_input)
    test_words = np.array(df_test.text.str.lower().values)

    test_words_binary_matrix = bag_of_words.transform(test_words)
    test_words_tfidf_matrix = tfidf.transform(test_words_binary_matrix)
    feature_names = bag_of_words.get_feature_names_out()

    save_matrix(df_test, test_words_tfidf_matrix, feature_names, test_output)


class Featurize(package1.Node):
    """Featurize the data set.

    Attributes
    ----------
    max_features : int
        Maximum number of features.
    ngrams : int
        Maximum ngrams.
    prepared : str
        Prepared data set.
    features : str
        Output features.
    """
    max_features: int = package1.params(200)
    ngrams: int = package1.params(2)

    prepared: str = package1.deps_path()
    features: str = package1.outs_path("data/features")

    def run(self):
        np.set_printoptions(suppress=True)

        train_input = os.path.join(self.prepared, "train.tsv")
        test_input = os.path.join(self.prepared, "test.tsv")
        train_output = os.path.join(self.features, "train.pkl")
        test_output = os.path.join(self.features, "test.pkl")

        os.makedirs(self.features, exist_ok=True)

        bag_of_words = CountVectorizer(
            stop_words="english",
            max_features=self.max_features,
            ngram_range=(1, self.ngrams),
        )
        tfidf = TfidfTransformer(smooth_idf=False)

        generate_and_save_train_features(
            train_input=train_input,
            train_output=train_output,
            bag_of_words=bag_of_words,
            tfidf=tfidf,
        )

        generate_and_save_test_features(
            test_input=test_input,
            test_output=test_output,
            bag_of_words=bag_of_words,
            tfidf=tfidf,
        )
