# download original data from https://datasets.cgv.tugraz.at/pattern-benchmark/

mkdir datasets
wget https://datasets.cgv.tugraz.at/pattern-benchmark-v0.1.zip
unzip -d datasets/pattern-benchmark-v0.1/ pattern-benchmark-v0.1.zip

# remove unnecessary files

rm pattern-benchmark-v0.1/*/*.ply
rm pattern-benchmark-v0.1/*/*.jpg
rm pattern-benchmark-v0.1/*/*.pat*