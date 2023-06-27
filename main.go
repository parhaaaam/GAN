package graphic_ml_project

import (
	"graphic_ml_project/gan"
	"graphic_ml_project/imageprocessing"
	"image/jpeg"
	"log"
	"os"

	"github.com/nfnt/resize"
)

func main() {
	// Open input image file
	inputFile, err := os.Open("input.jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer inputFile.Close()

	// Decode input image file
	inputImage, err := jpeg.Decode(inputFile)
	if err != nil {
		log.Fatal(err)
	}

	// Preprocess input image
	inputImage = imageprocessing.PreprocessImage(inputImage)

	// Generate output image using GAN model
	outputImage := gan.GenerateImage(inputImage, "gan_model.pb")

	// Postprocess output image
	outputImage = imageprocessing.PostprocessImage(outputImage)

	// Resize output image to 512x512
	outputImage = resize.Resize(512, 512, outputImage, resize.Lanczos3)

	// Save output image to file
	outputFile, err := os.Create("output.jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer outputFile.Close()

	// Encode output image to JPEG format and save to file
	if err := jpeg.Encode(outputFile, outputImage, nil); err != nil {
		log.Fatal(err)
	}
}
