import argparse
from documentprocessor import DocumentProcessor
from clustering import kmean_process, dbscan_process

from helpers import *


def create_parser():
    my_parser = argparse.ArgumentParser(description='This tool splits articles into number of clusters using ML'
                                                    ' algorithms')
    my_parser.add_argument('Path',
                           metavar='path',
                           type=str,
                           help='the path to directory containing articles')

    my_parser.add_argument('Clusters',
                           metavar='clusters',
                           type=int,
                           help='number of clusters needed')

    my_parser.add_argument('File_format',
                           metavar='file_format',
                           type=str,
                           help='file format of articles txt or pdf')

    my_parser.add_argument('-o',
                           '--options',
                           type=str,
                           help='specify what elements you want to use to split articles. t-title, a-abstract, '
                                'k-keywords, c-content. Example usage -o tac. Default option "tac"',
                           default='tac')

    my_parser.add_argument('-a',
                           '--algorithm',
                           type=str,
                           help='choose algorithm used to cluster articles> k-kmeans, d-dbscan. Default option kmeans',
                           default='k')

    return my_parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    validate_input_parameters(args)

    path = args.Path
    num_of_clusters = args.Clusters
    options = args.options
    algorithm = args.algorithm
    file_format = args.File_format

    if file_format == 'pdf':
        files = get_all_pdf_files(path)
        read_data_function = convert_pdf_to_txt
    else:
        files = get_all_txt_files(path)
        read_data_function = read_txt

    files = [os.path.join(path, i) for i in files]

    file_text = read_data_multithread(files, read_data_function, threads=1)

    all_words = get_zeroed_dictionary_with_all_worlds(file_text)

    # is required to evaluate idf
    world_presence_in_docs = get_word_presence_in_docs(file_text, all_words)

    x = []
    y = []
    for file, text in file_text:
        dp = DocumentProcessor(text, all_words, world_presence_in_docs, options)
        x.append(dp.get_processed_data())
        y.append(file)

    x = np.array(x)



    if 'k' in algorithm:
        result = kmean_process(x, y, num_of_clusters)
        save_result_to_file("kmeans", result)
    if 'd' in algorithm:
        result = dbscan_process(x, y, num_of_clusters)
        save_result_to_file("dbscan", result)



if __name__ == '__main__':
    main()
