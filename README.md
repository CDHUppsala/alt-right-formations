# Alt-Right Formations


## Requirements

The project uses `conda` for dependency management. First make sure conda is
installed, you can follow the regular install instructions for
[linux](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).
If on the server you can simply activate the environment by

```
conda activate /fullpath/to/env
```

Or for convenience fist `export
CONDA_ENVS_PATH=/path/to/projectfolder` and them simply
`conda activate 3.10.8`. If running somewhere else, first create the environment

```
conda env create -f environment.yml
```


## Data collections

Most of the configuration e.g. query, fields and start date for the data
collection are set in the `config.yaml`. The snippet below shows an example of
how to start collecting data.

``` bash
# example collect 1000 tweets
python data_collection/collect_tweets.py -t "your-beaerer-token" -n 1000
```

Data in pickle-format stored under `data_collection/data` and logs under
`data_collections/logs`. One can optionally add the argument `--next` to pass a
pagination token, specifying where the collection should start (see more:
<https://developer.twitter.com/en/docs/twitter-api/pagination>)

## Analysis

The directory `analysis` contains multiple scripts to generate different parts
of the analysis.

### `process_txt.py` 

This script processes the raw tweet text by:

1. Removing stopwords, symbols, numbers, punctuation and emojis
2. Generates lemmatized and stemmed tokens
3. Generates all bigrams resulting from nbr 2

### `timeline_plots_interactive.py`

Generate several interactive plots using `Plotly` of word occurrences across
the corpus. Plots are stored under `analysis/plots`.



