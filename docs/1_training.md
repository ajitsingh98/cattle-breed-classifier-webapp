# Training the Model

1.  Deep learning library
2.  Setting up Colab
3.  Getting the data
4.  Training the model 

## Deep learning library:  fastai
The [fastai](https://github.com/fastai/fastai) deep learning library, version 1.0.61 was utilized.  Fastai runs on top of PyTorch.   The [fastai MOOC](https://docs.fast.ai) was officially released to the public in early 2019.

## GPU:  Google Colab
The data was retrieved and analyzed on Google Colab Platform[(GCP)](https://colab.research.google.com/).  

For this project, I used a single 12GB NVIDIA Tesla K80 GPU.

## Dataset: Indian-Cattle-Breed-Images
The [Indian-Cattle-Breed-Images](https://drive.google.com/uc?export=download&id=1VSTdl6-AN701ER5mmu2MbO_V26a4C811) data was used which included **30 popular indian breed cattles categories** with a total of 4K images.  Thus, each class had around 150 images, of which all are manually reviewed test images:    
>Since all the images were collected through internet via several sources like *YouTube*, *Pinterest*, *Google Images* etc., the training images were not cleaned, and thus they contained large amount of noise. This comes mostly in the form of intense colors, low quality, blurr and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels.

#### Data Citation
Ajit Kumar Singh, IIT Guwahati, Indian-Cattle-Breed-Images - 2020


#### Retrieving the data
The dataset size is around 3GB and can be retrieved using:  
```bash
wget https://drive.google.com/uc?export=download&id=1VSTdl6-AN701ER5mmu2MbO_V26a4C811
```

## Training the data:  Resnet-50 CNN

### Training Time
I used the Resnet-50 CNN architecture.  The model took about 7 hours to run on Google Colab. 

### Training the Deep Learning Model
The code used for training the data is available in the repository [sajit9285/cattle-breed-classifier](https://github.com/sajit9285/cattle-breed-classifier-webapp) in the notebook section.

### Output from the Deep Learning Model
The output of the deep learning model is a file with weights.  The file is called `export.pkl` (or `model.pkl`).  If you train the model as in this repo, the model is saved to the `models` folder.  

The `export.pkl` file can be downloaded to your local computer from Jupyter.

The `export.pkl` file may be too large to be included in the git commit.  There are various options for proceeding with that size dataset:  
1.  Store the model on google drive.
2.  Store the model on GitHub releases.
3.  Store the model on a bucket in the cloud.  

I stored my final model data file on google drive: https://drive.google.com/uc?export=download&id=1VSTdl6-AN701ER5mmu2MbO_V26a4C811

 
