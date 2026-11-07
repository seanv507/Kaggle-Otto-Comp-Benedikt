# Word2Vec item embeddings

These are 50-d item embeddings per fold, from gensim's skip-gram Word2Vec on the fold's sessions. The GRU (`01d`) uses them as its initial item weights, and the rankers use them for similarity features.

```bash
uv run python 01a_FE_Word2Vec/word2vec.py --sample 0.05 --folds 0
uv run python 01a_FE_Word2Vec/word2vec.py             # all folds
uv run pytest 01a_FE_Word2Vec
```

| file | replaces | writes |
|---|---|---|
| `word2vec.py` | `000_Word2Vec.ipynb` | `data_folds/fold_<k>/word2vec.parquet`: (`aid` Int32, `emb` Array(Float32, 50)) |

Each session of `folds.fe_events(k)` is a sentence of its aids in time order, and every event counts, so repeats stay in. Benny's parameters: `vector_size=50, epochs=5, sg=1, window=3, sample=1e-3, ns_exponent=1, min_count=1`.

On 5% of fold 0 (728k sessions), the run takes 68 s and 1.4 GB of RAM. For aids whose top `caco` covisit weight is at least 20, the mean cosine to that partner is 0.88, against 0.47 to a random aid. The full data is about 20 times as big. gensim runs on the CPU only, so on the A100 pod the number of CPU cores sets the time.

## Differences from Benny's code

- **Output format:** parquet instead of a pickled dict `{aid: vector}` (`word2vec.emb`).
- **Corpus as a file:** the sentences go to gensim as a text file (`corpus_file`) rather than a Python list of 14.5M lists, which uses much less memory. Neither way is deterministic with several workers. `--workers` defaults to the number of CPUs; Benny used 8.
- **Sentence order:** sessions are in session order. Benny's came in the order of cuDF's `groupby(...).agg(list)`.
