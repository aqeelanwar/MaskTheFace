# MaskTheFace - Convert face dataset to masked dataset
![cover_photo](images/MaskTheFace.png)
## Frequently asked question (FAQ)

### 1. dlib ImportError: DLL load failed
If you are running MaskTheFace on windows, you might face dlib import error. You can solve this error by uninstalling dlib installed through requirements.txt and install from scratch. Use the following commands
```
pip uninstall dlib
git clone https://github.com/davisking/dlib.git
cd dlib
python setup.py install
```
This will build and install the dlib python library. Make sure you are in virtual environment you plan on using MaskTheFace in when you run the above commands.

### 2. fetch_dataset.py util import error
fetch_dataset.py is supposed to be run from the parent folder i.e.

```
# Correct
cd MaskTheFace
python utils/fetch_dataset.py

# Wrong
cd MaskTheFace/utils
python fetch_dataset.py
```