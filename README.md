
# DCGAN implementation using PyTorch


## Introduction
Deep Convolutional Generative Adversarial Network (DCGAN) is a type of generative model that utilizes deep convolutional neural network to generate synthetic images. It is a specifically modified version of GANs that uses strided convolutions and fractional-strided convolutions for discriminator and generator respectively. This project aims to implement DCGAN using PyTorch and leverage the power of deep learning to generate realistic MNIST digits.
## Dataset

MNIST stands for Mixed National Institute of Standards and Technology, which has produced a handwritten digits dataset. The MNIST dataset consists of 60,000 grayscale images of handwritten digits (0-9) with a size of 28x28 pixels. It is widely used as a benchmark dataset for various machine learning tasks, including image classification and generation.
## Installation

To run this implementation, follow these steps:
1. Clone this repository: 'git clone https://github.com/Narayan-21/Pytorch_DCGAN.git'
2. Install the required dependencies: 'pip install -r requirements.txt'
## Usage

1. Navigate to the project directory: 'cd your-repo'
2. If you want then change the hyperparameters that are otherwise predecided according to the original DCGAN paper, in the 'train.py' file.
3. Run the DCGAN: 
```python
python train.py
```


## Model Architecture
The DCGAN architecture consists of a generator network and a discriminator network. The generator network uses transposed convolutions to upsample random noise into meaningful images, while the discriminator network classifies the generated images as real or fake. For a detailed architecture diagram, refer to the image below:
[img]
Guidelines for stable DCGANs (as mentioned in the paper) are:
1. Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
2. Use batchnorm in both the generator and the discriminator.
3. Remove fully connected hidden layers for deeper architecture.
4. Use ReLU activation in generator for all layers except for the output, which uses Tanh.
5. Use LeakyReLU activation in the discriminator for all layers.
## Training Process
The DCGAN was trained for 5 epochs with a batch size of 128. The learning rate was set to 0.0002, and Adam optimizer and Binary Cross-Entropy loss function was used. Additionally, batch normalization and ReLU activation were applied to stabilize the training process.
## Results
To dive into the results, follow these steps:
1. Navigate to the project directory: 'cd your-repo' in command prompt.
2. Now run the tensorboard using following command:
```python
tensorboard --logdir logs
```
( Make sure to use this command only after you have created the logs folder, that is after running the train.py file)

3. Make sure to open and analyze the tensorboard on your web browser on localhost using the following default url: 'http://localhost:6006/'
## License

This Project is licensed under [Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/).


## Acknowledgements

 - [The PyTorch team for their excellent deep learning library.](https://pytorch.org/)
 - [The authors of the original DCGAN paper: Radford, Alec, Luke Metz, and Soumith Chintala.](https://arxiv.org/pdf/1511.06434.pdf)


## Contact Information
For any questions or feedback, please contact me at naaidjan.19@gmail.com.