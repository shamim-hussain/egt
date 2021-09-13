echo 'Downloading dataset ...'
wget -O datasets.tar.gz https://zenodo.org/record/5500978/files/datasets.tar.gz?download=1
echo 'Extracting dataset ...'
tar -xvf datasets.tar.gz
rm datasets.tar.gz
echo 'Removed datasets.tar.gz'
echo 'Done!'