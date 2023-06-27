package imageprocessing

import (
	"image"
	"image/draw"

	"github.com/nfnt/resize"
)

func PostprocessImage(outputImage image.Image) image.Image {
	// Resize output image to 256x256
	resizedImage := resize.Resize(256, 256, outputImage, resize.Lanczos3)

	// Convert output image to RGBA format
	outputImageRGBA := image.NewRGBA(resizedImage.Bounds())
	draw.Draw(outputImageRGBA, resizedImage.Bounds(), resizedImage, image.Point{}, draw.Src)
	return outputImageRGBA
}
