import os

def get_all_pdf_files(path):
    files = os.listdir(path)
    files = [f for f in files if f.endswith('.pdf')]
    return files
