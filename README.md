
## FUJISAN
FUJISAN is a tool for predicting functional identity based on similarities of protein pairs between full-length sequences, between domain structures, and between pocket-forming residues.


## Download & Installation
```bash
git clone https://github.com/sfujita0601/FUJISAN.git
cd FUJISAN

python3.11 -m venv --system-site-packages --clear \
        --upgrade-deps $(pwd)/.venv
source .venv/bin/activate

python3.11 -m pip install git+https://github.com/optuna/optuna.git
python3.11 -m pip install optuna-integration
python3.11 -m pip install shap
python3.11 -m pip install ipython
python3.11 -m pip install .

```

## Demo


An example to run:
```bash
python3.11 FUJISAN/run_FUJISAN.py
```

Expected output files:
- Analysis/ : .txt files and .csv files.
- Figure/ : Figures used in the article.   
