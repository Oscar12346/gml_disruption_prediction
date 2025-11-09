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

Run E-STFGNN model:

```shell
python run_advanced_novel.py
```

Run ablation study for E-STFGNN model with different weather feature settings:

```shell
python run_ablation_study.py --weather_features=all
python run_ablation_study.py --weather_features=none
python run_ablation_study.py --weather_features=continuous
python run_ablation_study.py --weather_features=boolean
```
