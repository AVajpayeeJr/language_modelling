# language_modelling

Training scripts for n-gram and neural language models -

N-Gram ([SRILM](http://www.speech.sri.com/projects/srilm/)):
1. Word N-Gram 
2. Word + Class N-Gram

Neural ([Keras](https://keras.io/)):
1. Word RNN
2. Char RNN
3. Word + Char RNN
4. Class Factorized Word RNN

## Requirements

- Keras-2.2.4
- SRILM-1.7.2
- [Percy Liang's Threaded Brown Clustering](https://github.com/ajaech/brown-cluster)

## Usage

Updated SRILM and Brown-Clustering Paths in sourcefile.

```bash
source sourcefile
```

Pre-Process a corpus file to make train/val/test files and perform Brown Clustering. Intermediate files created at <LANGUAGE_CODE> directory. 
```python
python3 dataset_preparation.py --source_monolingual_file <FILE_PATH> --language <LANGUAGE_CODE> --config config.yaml --cluster --num_clusters 1000 --clustering_home_path <PATH_TO_BROWN_DIR>
```

Run N-Gram Language Models through SRILM. 
```bash
./ngram_lm.sh <LANGUAGE_CODE> <OUTPUT_BASE_DIR> <NUM_CLUSTERS> <MIXING LAMBDA>
```

Run Neural Language Models. 
```python
python3 neural_lm --language <LANGUAGE_CODE> --config config.yaml --type <'word'/'char'/'word_char'/'word_class'> --output_base_dir <DIR_PATH>
```

## Things to Do

1. Test Neural Word+Class Based LM and write code for evaluating perplexity.
2. Write methods for rescoring lattice 
3. NNLM to Approximate N-Gram LM
4. Interpolating NNLM with N-Gram LM

## References
- [Class+Word Factorized RNNLM](https://link.springer.com/article/10.1186/1687-4722-2013-22) (Implemented. To test and add functionality)
- [RNNLM to WFST](https://pdfs.semanticscholar.org/38db/ee3085539c0d0c61f6722b9d198a4a921b92.pdf)  (Maybe to implement)
