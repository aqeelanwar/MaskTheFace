# MaskTheFace - Convert face dataset to masked dataset
![cover_photo](images/MaskTheFace.png)

## What is MaskTheFace?
MaskTheFace is computer vision based scripts to mask faces in images. It provides a number of masks to select from.  
It is difficult to collect mask dataset under various conditions. MaskTheFace can be used to convert any existing face dataset to masked-face dataset.
MaskTheFace identifies all the faces within an image, and applies the user selected masks to them taking into account various limitations such as fit, color and brightness variations etc.  
A single image, or entire directory of images can be used as input to code.
![cover_photo](images/example1.png)
## How to install MaskTheFace
It’s advisable to [make a new virtual environment](https://towardsdatascience.com/setting-up-python-platform-for-machine-learning-projects-cfd85682c54b) and install the dependencies. Following steps can be taken to download get started with MaskTheFace
### Clone the repository
```
git clone https://github.com/aqeelanwar/MaskTheFace.git
```

### Install required packages
The provided requirements.txt file can be used to install all the required packages. Use the following command

```
cd MaskTheFace
pip install –r requirements.txt
```

This will install the required packages in the activated python environment.
