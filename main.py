from helpers import *


def main():

    files = get_all_pdf_files(path)

    file_text = []
    for file in files:
        f = os.path.join(path, file)
        text = convert_pdf_to_txt(f)
        file_text.append((file, text))

    all_words = get_zeroed_dictionary_with_all_worlds(file_text)
    #print(all_words)


if __name__ == '__main__':
    main()
