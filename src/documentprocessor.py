import numpy as np

from helpers import clear_text_and_change_to_vector


class DocumentProcessor:
    """
    class which transform article text into features(required to train clustering model)
    """

    def __init__(self, text, all_words, words_presence_in_docs, options='n'):
        self.all_words = all_words
        self.text_vec = clear_text_and_change_to_vector(text)
        self.words_presence_in_docs = words_presence_in_docs
        self.processed = []

        # TODO handle errors, return name of failing file and go on with processing
        # k-key t-title a-abstract c-content
        if 'k' in options:
            self.if_keywords_present()
            self.append_data(self.get_key_feature())
        if 't' in options:
            self.append_data(self.get_title_feature())
        if 'a' in options:
            self.append_data(self.get_abstract_feature())
        if 'c' in options:
            self.append_data(self.get_content_feature())
        if 'n' in options:
            self.append_data(self.get_tfidf_feature())

    def if_keywords_present(self):
        if 'keyword' in self.text_vec:
            return True
        else:
            return False

    def get_feature_vector(self, words):
        """
        create and returns vector of features(list containing ordered word counts based on zeroed all_word dictionary)

        :param words: list of words
        :return: list of numbers representing word counts in passed argument
        """
        dictionary = self.all_words.copy()
        for i in words:
            dictionary[i] += 1

        return dictionary.values()

    def get_feature_dict(self, words):
        """
        create and returns dictionary containing word counts based on passed as argument list of words
        and zeroed all_word dictionary

        :param words: list of words
        :return: dictionary with word counts
        """
        dictionary = self.all_words.copy()
        for i in words:
            dictionary[i] += 1

        return dictionary

    def get_key_feature(self):
        """
        gets all words between 'keyword' and 'introduct' and transforms it into features vector

        :return:
        """
        if self.if_keywords_present():
            start = self.text_vec.index('keyword') + 1
            stop = self.text_vec.index('introduct')
            words = self.text_vec[start:stop]
            return self.get_feature_vector(words)
        else:
            raise Exception('Despite keywords option selected, keywords are not present in all pdfs')

    def get_title_feature(self):
        """
        gets all words up to 'abstract' and transforms it into features vector

        :return:
        """
        start = 0
        stop = self.text_vec.index('abstract')
        words = self.text_vec[start:stop]
        return self.get_feature_vector(words)

    def get_abstract_feature(self):
        """
        gets all words between 'abstract' and 'introduct' or 'keywords' and transforms it into features vector

        :return:
        """
        start = self.text_vec.index('abstract') + 1
        if self.if_keywords_present():
            stop = self.text_vec.index('keyword')
        else:
            stop = self.text_vec.index('introduct')

        words = self.text_vec[start:stop]
        return self.get_feature_vector(words)

    def get_content_feature(self):
        """
        gets all words after 'introduct' and transforms it into features vector, creates tf idf features vector


        :return:
        """
        start = self.text_vec.index('introduct') + 1
        stop = -1
        words = self.text_vec[start:stop]

        return self.get_feature_vector(words)

    def get_tfidf_feature(self):
        """
                gets all words after 'introduct' and transforms it into features vector, creates tf idf features vector


                :return:
                """
        start = self.text_vec.index('introduct') + 1
        stop = -1
        words = self.text_vec[start:stop]

        feature_dict = self.get_feature_dict(words)

        tfidf = []
        length = len(words)
        for i in feature_dict.keys():
            tf = feature_dict[i] / length
            idf = np.log(feature_dict[i] / self.words_presence_in_docs[i] + 1)
            tfidf.append(tf * idf)

        return tfidf

    def get_processed_data(self):
        return self.processed

    def append_data(self, words_values):
        for i in words_values:
            self.processed.append(i)
