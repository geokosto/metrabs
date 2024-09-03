

--------------------------!!!!!!!!!!!!!!!!!!!!!!!-----------------------

conda create --name "metrabs"
conda create --name "metrabs_env"

conda activate metrabs

cd rlemasklib
conda install numpy
python setup.py build_ext --inplace
python setup.py install
cd ..

```bash
conda install cachetools cython ezc3d ffmpeg imageio matplotlib mkl numba numpy pandas Pillow scikit-image scikit-learn tqdm -c conda-forge
```

```bash
pip install -r requirements.txt
```