https://zenodo.org/records/12636070

pip install kaggle
mkdir -p data/m5
chmod 600 ~/.kaggle/kaggle.json

kaggle competitions download \
  -c m5-forecasting-accuracy \
  -p data/m5

cd data/m5
unzip -o m5-forecasting-accuracy.zip
cd ../..


