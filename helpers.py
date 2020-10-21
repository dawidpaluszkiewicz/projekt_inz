import os
import sys
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from nltk.stem import PorterStemmer
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from multiprocessing.dummy import Pool as ThreadPool


def read_txt(path):
    tmp = []
    with open(path) as f:
        for line in f:
            tmp.append(line)
    return path, ''.join(tmp)


def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()

    return path, text


def read_data_multithread(files, reader_function, threads=8):
    pool = ThreadPool(threads)
    results = pool.map(reader_function, files)
    pool.close()
    pool.join()
    return results


def get_stop_words_list():
    """
    Reads stop word list from file "stopwords" into list and returns it.

    :return: list of stop words
    """
    stop_words = []
    with open('stopwords') as f:
        for line in f:
            tmp = line.replace('\n', '')
            stop_words.append(tmp)

    return stop_words


STOP_WORDS = get_stop_words_list()


def clear_text_and_change_to_vector(text):
    """
    Clears text. Remove all redundant signs, remove stop words, stem words and transform text into vector of words.

    :param text: text to process
    :return: list of cleared and transformed words
    """
    to_remove = list('1234567890,.()-!@#$%^&*_+={}[];:/—')
    for sign in to_remove:
        text = text.replace(sign, ' ')

    text = text.replace('   ', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('\n', ' ')
    word_vec = text.split(' ')
    word_vec = [i.strip() for i in word_vec]
    word_vec = [i for i in word_vec if len(i) > 1]
    word_vec = [i.lower() for i in word_vec]
    word_vec = [i for i in word_vec if i not in STOP_WORDS]
    ps = PorterStemmer()
    word_vec = [ps.stem(i) for i in word_vec]

    return word_vec


def get_all_pdf_files(path):
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.pdf')]

    if len(files) == 0:
        raise FileNotFoundError("There is no any of pdf articles")

    return files


def get_all_txt_files(path):
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.txt')]

    if len(files) == 0:
        raise FileNotFoundError("There is no any of txt articles")

    return files


def get_zeroed_dictionary_with_all_worlds(file_text):
    """
    Concatenates all documents to process. Creates vector with all words appearing in articles.
    Creates and returns a dictionary containing all unique processed words.

    :param file_text: list of tuples containing name of the file and its text
    :return: dictionary containing all unique words included in all articles
    """
    texts = []
    for _, text in file_text:
        texts.append(text)

    combined_text = ' '.join(texts)
    word_vec = clear_text_and_change_to_vector(combined_text)

    all_words = {}
    for i in word_vec:
        all_words[i] = 0

    return all_words


def get_word_presence_in_docs(path_text, all_words):
    """
    For each word included in all_word dictionary it assigns number of documents the specific word is present at least
    once.

    :param path_text: list of tuples containing name of the file and its text
    :param all_words: zeroed dictionary with all unique words included in all articles
    :return: dictionary containing word count in all documents
    """
    texts = []
    for _, text in path_text:
        texts.append(clear_text_and_change_to_vector(text))

    word_presence = all_words.copy()
    for i in all_words.keys():
        count = 0
        for text in texts:
            if i in text:
                count += 1
        word_presence[i] = count

    return word_presence


def save_result_to_file(file, result):
    with open(file, 'w') as f:
        for i in result:
            print(f'{i[0]} {i[1]}', file=f)


def validate_input_parameters(params):  # Maybe to develop further to make gui app return proper error
    num_of_clusters = params.Clusters
    file_format = params.File_format

    if num_of_clusters < 2:
        print("Number of clusters must be greater than 1")
        sys.exit(1)

    if file_format not in ['pdf', 'txt']:
        print("Wrong file format parameter")
        sys.exit(1)


def plot_PCA(x, centres):
    """
        Reduces number of data dimensions to 2 most relevant one and plots it along with cluster centers.


        :param x: data representation of articles
        :param centres: centres of clusters
        """
    num_of_elements = len(x)
    num_of_clusters = len(centres)
    x = [i[1] for i in x]
    for i in range(num_of_clusters):
        x.append(centres[i])

    x = np.array(x)
    x_norm = (x - x.min()) / (x.max() - x.min())
    pca = PCA(n_components=2)  # 2-dimensional PCA
    transf = pd.DataFrame(pca.fit_transform(x_norm))

    for i in range(num_of_clusters):
        plt.scatter(transf[0][(num_of_elements // num_of_clusters) * i:(num_of_elements // num_of_clusters) * (i + 1)],
                    transf[1][(num_of_elements // num_of_clusters) * i:(num_of_elements // num_of_clusters) * (i + 1)],
                    label=str(i))

        plt.scatter(transf[0][num_of_elements + i], transf[1][num_of_elements + i], label='{} cluster center'.format(i),
                    s=150)

    plt.legend()
    plt.show()
