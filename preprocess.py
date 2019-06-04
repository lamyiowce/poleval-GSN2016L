import argparse
import multiprocessing
import pickle
import os.path

from utils import CorpusPreprocessor


def main():
    parser = argparse.ArgumentParser(description='Text preprocessing.')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--ncpus', default=multiprocessing.cpu_count(), type=int)
    args = parser.parse_args()

    if os.path.isfile(args.out) and not args.overwrite:
        print("File " + args.out + " already exists. Set flag --overwrite to overwrite.")
    else:
        with multiprocessing.Pool(args.ncpus) as pool:
            filename = args.data
            lines = open(filename, encoding='utf-8').read().strip().split('\n')

            corpus_preprocessor = CorpusPreprocessor()
            lines = pool.map(corpus_preprocessor.transform_text, lines)
            corpus_preprocessor.fit_corpus(lines)
            data = pool.map(corpus_preprocessor.mask_text, lines)

        # data = list(map(
        #     lambda sentence, word, x:
        #     (
        #         (sentence, word, x),
        #         (corpus_preprocessor.numerize_sentence(sentence), corpus_preprocessor.numerize_word(word), x)
        #     ),
        #     masked
        # ))

        with open(args.out, "wb+") as file:
            pickle.dump(data, file)


if __name__ == '__main__':
    main()
