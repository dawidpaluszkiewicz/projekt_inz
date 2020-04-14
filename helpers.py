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


def get_all_pdf_files(path):
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.pdf')]
    return files
