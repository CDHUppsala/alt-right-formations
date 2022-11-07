# Alt-Right Formations


## Requirements

The project uses `poetry` for dependency management. Make sure to have it
installed and do the following to activate the virtual environment.

``` bash
poetry install
poetry shell
```

## Data collections

Most of the configuration e.g. query, fields and start date for the data
collection are set in the `config.yaml`. The snippet below shows an example of
how to start collecting data.

``` bash
# example collect 1000 tweets
python data_collection/collect_tweets.py -t "your-beaerer-token" -n 1000
```

Data in csv-format stored under `data_collection/data` and logs under
`data_collections/logs`. One can optionally add the argument `--next` to pass a
pagination token, specifying where the collection should start (see more:
<https://developer.twitter.com/en/docs/twitter-api/pagination>)
