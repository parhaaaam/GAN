# Generative Adversarial Network (GAN) for High-Resolution Image Generation

This project is a Go implementation of a Generative Adversarial Network (GAN) for generating high-resolution images of human faces. The GAN is trained on a dataset of human faces and can generate new, synthetic images that resemble the training data.

## Dependencies
    GoCV
    TensorFlow
    resize
    image

## Usage

To use this GAN to generate high-resolution images of human faces, follow these steps:

1. Clone this repository to your local machine.
2. Install the dependencies listed above.
3. Prepare a dataset of human faces for training the GAN. The dataset should be a
   collection of images in JPEG or PNG format.
4. Train the GAN on the dataset using the `train` command.
   The command takes the path to the dataset as input and saves the trained GAN model to a 
   file.

```azure
go run train.go --dataset /your/dataset --model gan_model.pb
```
5.Use the trained GAN model to generate high-resolution images
  of human faces using the `generate` command. The command takes
  the path to an input image and the path to the trained GAN 
  model as input, and saves the generated image to a file.
```azure
go run generate.go --input input.jpg --model gan_model.pb --output output.jpg
```

## Package

The `gan` package contains the implementation of the GAN and
the `imageprocessing` package contains the implementation of
the image preprocessing and postprocessing functions.

### `gan` package

`train.go`: Trains the GAN on a dataset of human faces and saves the
trained model to a file.

`generate.go`: Generates a high-resolution image of a human face using a pretrained GAN model.

### `imageprocessing` package

`preprocess.go`: Preprocesses an input image to extract the face
region and resize it to a standard size.

`postprocess.go`: Postprocesses an output image to resize it to
the desired size.

