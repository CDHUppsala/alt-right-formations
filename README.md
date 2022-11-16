# Alt-Right Formations


## Requirements

The project uses `conda` for dependency management. First make sure conda is
installed, you can follow the regular install instructions for
[linux](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).
If on BEAST you can simply activate the environment by

```
conda activate /zpool/beast-mirror/alt-right-formations/3.10.8
```

Or for convenience fist `export
CONDA_ENVS_PATH=/zpool/beast-mirror/alt-right-formations/` and them simply
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

Data in csv-format stored under `data_collection/data` and logs under
`data_collections/logs`. One can optionally add the argument `--next` to pass a
pagination token, specifying where the collection should start (see more:
<https://developer.twitter.com/en/docs/twitter-api/pagination>)
