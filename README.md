# CycleGAN-Photo-to-Monet-Painting

## Introduction
CycleGAN-based deep learning project for transforming photos into Monet-style paintings, showcasing custom architecture and extensive experimentation in Python with TensorFlow Keras.

### Dataset Distribution
* 7038 regular images in JPG format
* 300 Monet paintings in JPG format

This project was done as a part of a Kaggle Style Transfer Competition. You can find more information about the competition and get the dataset [here](https://www.kaggle.com/competitions/gan-getting-started/overview).

### What is CycleGAN?

CycleGAN is a type of Generative Adversarial Network (GAN) designed for unpaired image-to-image translation. Unlike traditional methods that require paired examples of images in different styles for training, CycleGAN learns to translate images between two domains without any direct correspondences.

### Photo to Monet Style Transfer

The task of transforming photographs into Monet-style paintings is an example of unpaired image-to-image translation. In this case, we aim to convert everyday photographs into images that capture the distinct visual characteristics of Monet's paintings, such as vibrant colors, textured brushstrokes, and dreamy atmospheres.

CycleGAN achieves this transformation by simultaneously training two generators and two discriminators. The generators learn to translate images from one domain (photographs) to another (Monet-style paintings), while the discriminators distinguish between real and generated images. Through adversarial training and cycle-consistency loss, CycleGAN ensures that the translated images not only resemble the target style but also preserve the content of the original photographs.

### How to Use

In the project's files, you can find:
1. **Google Colab Notebooks** (inside the `notebooks` folder):
    * **ADL_Test_Notebook.ipynb**: Provide an image as input and get a Monet-style output.
    * **ADL_Training_Notebook.ipynb**: Learn about the training process and the different approaches that were tested.
2. **Python Scripts** (inside the `scripts` folder):
    * **ADL_Test_Script.py**: Python script version of the test notebook.
    * **ADL_Training_Script.py**: Python script version of the training notebook.

Both notebooks and scripts contain a step-by-step guide on how to use them.

### Trained Models
The trained models used for this project are also included in the repository. You can use these models directly without retraining.
