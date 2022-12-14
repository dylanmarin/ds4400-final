# ds4400-final: Image Caption Generation
This is Dylan Marin and Julia Geller's final project
All of the code required to extract features using VGG and to train models is present. However, we made use of saving and loading our trained models. These models and feature files were unable to be uploaded, so there are links to download our entire models folder via Google Drive below. You 

Note about running the code: 
This code was run with Python 3.7.15 with Tensorflow 2.7.0 and Keras 2.7.0 and Cuda 11.6

## File structure:
Below describes our file structure and the purpose of each file

## src/
This folder contains our source code.

### common.py
    this file contains extracted utility methods that were used throughout the project at different points

### Logistic Decoder.py
    this file contains utils and the class pertaining to our Logistic Decoder model

### RNNDecoder.py
    this file contains utils and the class for both of our RNN models

### train_test_split.ipynb
    this file was used to split up the kaggle data into 3 separate sets (train, test, validation) and write those separated captions to a file

### image_feature_extraction.ipynb
    this file was created to use pre-trained VGG to extract and store features from ALL of our images. they were extracted to a file called 8k_features.pkl, which is linked below because it was too large to upload

### logistic_model.ipynb
    initially was used to define and test our entire logistic decoder class, but is now just used to show an example of how to use the Logistic Decoder class

### rnn_model.ipynb
    initially was used to define and test both of our RNN model classes, but is now just used to show an example of how to use each of them

### model_tuning.ipynb
    this file is where all of the models that were using for model selection, hyperparameter tuning, and testing were defined and tested. the lines used to train and save the models are commented out because they were already saved once. these files were also too large to upload. feel free to train the models yourself, or use the link below to download the pretrained models that we used. they should be extracted into a directory called models/ which is at the same level as source

## data
This folder is where we stored our data. 

### data/generated_captions.csv
    this file contains the captions generated by our last 3 models for all images in the test set

### data/flickr_8k
    This folder contains an images directory containing the jpg files of all images in the dataset. It also contains a captions.txt which is the raw captions for all of the images in the dataset. We also defined and extracted our train/test/validation sets to csv files in this directory. In addition to those 3, we defined a small_train.csv containing half of the training samples, and a train_and_val.csv containing the training samples in addition to the validation samples. This directory is also where the 8k_features.pkl will need to be extracted to.

### data:
    kaggle: https://www.kaggle.com/datasets/adityajn105/flickr8k?select=captions.txt
    extracted_features: https://drive.google.com/file/d/1U7SmHFVpccIjFDVdCO42gXNquHeFlIOx/view?usp=sharing
    models: https://drive.google.com/drive/folders/1Mo4PKAL_zx9v1o-0KZvThNtUQlv29yV8?usp=sharing
