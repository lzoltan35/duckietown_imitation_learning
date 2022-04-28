#! /bin/bash
set -e

conda remove --name duckietown --all -y
echo "Creating conda environment: duckietown"
echo '...'
conda env create -f conda_environment.yml

echo "Cloning gym-duckietown"
echo "..."
rm -rf gym-duckietown
git clone --branch v5.0.16 --single-branch --depth 1 https://github.com/duckietown/gym-duckietown.git ./gym-duckietown

echo "Installing gym-duckietown"
echo "..."
conda run -vn duckietown pip install -e gym-duckietown

echo "Copy custom maps to the installed Duckietown packages"
echo "..."
conda run -vn duckietown python -m utils.copy_maps

echo "Setup successful!"
