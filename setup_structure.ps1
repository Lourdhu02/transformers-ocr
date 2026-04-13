# Run from D:\Dev\transformers-ocr

Move-Item augment.py   engine\augment.py
Move-Item codec.py     engine\codec.py
Move-Item dataset.py   engine\dataset.py
Move-Item loss.py      engine\loss.py
Move-Item preprocess.py engine\preprocess.py

Move-Item svtr.py      models\svtr.py

Move-Item config.py    configs\config.py

Move-Item export_onnx.py tools\export_onnx.py

Move-Item colab_train.ipynb notebooks\colab_train.ipynb

New-Item engine\__init__.py    -ItemType File -Force
New-Item models\__init__.py    -ItemType File -Force
New-Item configs\__init__.py   -ItemType File -Force
New-Item tools\__init__.py     -ItemType File -Force

New-Item data\.gitkeep    -ItemType File -Force
New-Item weights\.gitkeep -ItemType File -Force
New-Item exports\.gitkeep -ItemType File -Force
New-Item logs\.gitkeep    -ItemType File -Force
