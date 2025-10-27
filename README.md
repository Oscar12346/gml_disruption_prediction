Create and activate a virtual environment:

```shell
python3 -m venv .venv
source ./.venv/bin/activate
```

Install required dependencies:

```shell
pip install -r requirements.txt
```

Preprocess all raw data:

```shell
python preprocess.py
```

Draw graph:

```shell
python plot.py
```
