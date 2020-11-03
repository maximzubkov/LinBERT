## LinBERT

To start training run the following commands from source dir:
```(bash)
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install -e .
sh data/download_espiranto.sh
python train.py
```