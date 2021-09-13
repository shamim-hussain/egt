echo 'Downloading dataset ...'
wget -O pcqm4m.tar.gz https://zenodo.org/record/5501020/files/pcqm4m.tar.gz?download=1
echo 'Extracting dataset ...'
tar -xvf pcqm4m.tar.gz
mkdir -pv datasets
mv PCQM4M datasets/
echo 'Moved to datasets/PCQM4M/'
rm pcqm4m.tar.gz
echo 'Removed pcqm4m.tar.gz'
echo 'Done!'