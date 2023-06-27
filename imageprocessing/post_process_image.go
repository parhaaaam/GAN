package imageprocessing

import (
	"image"

	"gocv.io/x/gocv"
)

func PreprocessImage(inputImage image.Image) image.Image {
	// Convert input image to Mat format
	matImage, err := gocv.ImageToMatRGBA(inputImage)
	if err != nil {
		panic(err)
	}

	classifier := gocv.NewCascadeClassifier()
	defer classifier.Close()
	if !classifier.Load("haarcascade_frontalface_default.xml") {
		panic("Error loading face detection classifier")
	}

	rects := classifier.DetectMultiScale(matImage)
	if len(rects) == 0 {
		panic("No faces detected in input image")
	}

	// Extract first face region and resize to 64x64
	rect := rects[0]
	faceImage := matImage.Region(rect)
	gocv.Resize(faceImage, &faceImage, image.Point{64, 64}, 0, 0, gocv.InterpolationLinear)

	// Convert face image to RGBA format
	faceImageRGBA := gocv.NewMat()
	defer faceImageRGBA.Close()
	gocv.CvtColor(faceImage, &faceImageRGBA, gocv.ColorBGRToRGBA)
	return faceImageRGBA.ToImage()
}
