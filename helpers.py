import os
from nltk.stem import PorterStemmer
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO


def convert_pdf_to_txt(path):  # TODO add exception handling
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
    return text


def get_stop_words_list():
    stop_words = []
    with open('stopwords') as f:
        for line in f:
            tmp = line.replace('\n', '')
            stop_words.append(tmp)

    return stop_words


STOP_WORDS = get_stop_words_list()


def clear_text_and_change_to_vector(text):
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
    return files


def get_zeroed_dictionary_with_all_worlds(file_text):
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
    texts = []
    for _, text in path_text:
        texts.append(clear_text_and_change_to_vector(text))

    result = all_words.copy()
    for i in all_words.keys():
        count = 0
        for text in texts:
            if i in text:
                count +=1
        result[i] = count

    return result
