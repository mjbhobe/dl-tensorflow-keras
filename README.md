# dl-tensorflow-keras
Contains iPython Notebooks for my articles on Medium.com. Most of these notebooks have been designed to run on Google Colab (https://colab.research.google.com/) as well as offline. Google Colab provides an IPython Notebook like environment pre-loaded with all scientific & deep-learning libraries + __a free GPU!__ That is motivation enough for me to use Colab for all my Deep Learning projects. All you need to use Google Colab is a Google account, which I'm sure you will have ;).

## Pre-requisites
You will need the following Python libraries (all pre-installed on Colab!)
```
numpy pandas matplotlib seaborn tensorflow keras 
```
Code is not written for any specific version of the libraries - I used whatever was available on Colab.

## Contents
1. **MNIST - Multiclass Classification - CNN - Keras.ipynb** :- MNIST digits classification with a Keras Convolutional Neural Network. Achieves 99% accuracy on test data!
2. 
I also provide some helper functions to load & save keras models + plot model's performance from training. These functions are available in the `kr_helper_funcs.py` 

## How to use
1. Download the repository from Github to a folder on your disk
2. If using Google Colab, upload all the notebooks (`*.ipynb`) and Python modules (`*.py`) files to your Google Drive  to a folder called `Colab Notebooks`
3. If using offline: start IPython Notebook and browse to folder where you downloaded these notebooks and use as usual. **NOTE:** Please set the USE_COLAB=False at the very beginning of the notebook.
