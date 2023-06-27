package gan

import (
	"graphic_ml_project/imageprocessing"
	"image"

	"github.com/disintegration/imaging"
	"github.com/nfnt/resize"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
)

func GenerateImage(inputImage image.Image, modelPath string) image.Image {
	// Load generator model
	model, err := loadModel(modelPath)
	if err != nil {
		panic(err)
	}

	inputImage = imageprocessing.PostprocessImage(inputImage)

	outputImage := generate(model, inputImage)

	outputImage = imageprocessing.PostprocessImage(outputImage)

	return outputImage
}

func loadModel(modelPath string) (*tensorflow.SavedModel, error) {
	// Load generator model from file
	model, err := tensorflow.LoadSavedModel(modelPath, []string{"generator"}, nil)
	if err != nil {
		return nil, err
	}

	return model, nil
}

func generate(model *tensorflow.SavedModel, inputImage image.Image) image.Image {
	// Convert input image to tensor
	inputTensor, err := imageToTensor(inputImage)
	if err != nil {
		panic(err)
	}

	outputTensor, err := runModel(model, inputTensor)
	if err != nil {
		panic(err)
	}

	outputImage, err := tensorToImage(outputTensor)
	if err != nil {
		panic(err)
	}

	return outputImage
}

func imageToTensor(inputImage image.Image) (*tensorflow.Tensor, error) {
	// Resize input image to 64x64
	inputImage = resize.Resize(64, 64, inputImage, resize.Lanczos3)

	// Convert input image to tensor
	inputTensor, err := imaging.Encode(inputImage, imaging.JPEG)
	if err != nil {
		return nil, err
	}
	inputTensorShape := []int64{1, int64(len(inputTensor))}
	inputTensorBytes := make([]byte, len(inputTensor))
	copy(inputTensorBytes, inputTensor)
	inputTensor, err = tensorflow.NewTensor(inputTensorBytes, inputTensorShape)
	if err != nil {
		return nil, err
	}

	return inputTensor, nil
}

func runModel(model *tensorflow.SavedModel, inputTensor *tensorflow.Tensor) (*tensorflow.Tensor, error) {
	// Run generator model with input tensor
	outputTensor, err := model.Session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			model.Graph.Operation("generator/input").Output(0): inputTensor,
		},
		[]tensorflow.Output{
			model.Graph.Operation("generator/output").Output(0),
		},
		nil,
	)
	if err != nil {
		return nil, err
	}

	return outputTensor[0], nil
}

func tensorToImage(outputTensor *tensorflow.Tensor) (image.Image, error) {
	// Convert output tensor to image
	outputTensorBytes, err := outputTensor.Value().([]byte)
	if err != nil {
		return nil, err
	}
	outputImage, err := imaging.Decode(outputTensorBytes)
	if err != nil {
		return nil, err
	}

	return outputImage, nil
}
