package gan

import (
	"fmt"
	"image"
	"image/jpeg"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/nfnt/resize"
	tensorflow "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func Train(datasetPath string, modelPath string) {
	// Set random seed for reproducibility
	rand.Seed(time.Now().UnixNano())

	// Load dataset
	dataset, err := loadDataset(datasetPath)
	if err != nil {
		panic(err)
	}

	generator, discriminator := createNetworks()

	generatorLoss, discriminatorLoss := createLossFunctions()

	optimizer := createOptimizer()

	graph := tensorflow.NewGraph()
	session, err := tensorflow.NewSession(graph, nil)
	if err != nil {
		panic(err)
	}
	defer session.Close()

	if err := session.Run(nil, map[tf.Output]*tf.Tensor{
		graph.Operation("init").Output(0): nil,
	}, nil); err != nil {
		panic(err)
	}

	// Train GAN
	for i := 0; i < 10000; i++ {
		batch := selectBatch(dataset, 16)

		fakeImages := generateImages(generator, batch)

		realLabels := createLabels(len(batch), 0.9)
		fakeLabels := createLabels(len(batch), 0.1)
		discriminatorLossValue, err := trainDiscriminator(discriminator, optimizer, batch, realLabels, fakeImages, fakeLabels, session, graph)
		if err != nil {
			panic(err)
		}

		generatorLossValue, err := trainGenerator(generator, discriminator, optimizer, batch, session, graph)
		if err != nil {
			panic(err)
		}

		if i%100 == 0 {
			fmt.Printf("Step %d: Generator loss = %f, Discriminator loss = %f\n", i, generatorLossValue, discriminatorLossValue)
		}
	}

	if err := saveModel(generator, modelPath); err != nil {
		panic(err)
	}
}

func loadDataset(datasetPath string) ([]image.Image, error) {
	// Load JPEG images from dataset directory
	var dataset []image.Image
	err := filepath.Walk(datasetPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && (filepath.Ext(path) == ".jpg" || filepath.Ext(path) == ".jpeg") {
			file, err := os.Open(path)
			if err != nil {
				return err
			}
			defer file.Close()
			image, err := jpeg.Decode(file)
			if err != nil {
				return err
			}
			dataset = append(dataset, image)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	// Resize images to 64x64
	for i, image := range dataset {
		dataset[i] = resize.Resize(64, 64, image, resize.Lanczos3)
	}

	return dataset, nil
}

func createNetworks() (*tf.Graph, *tf.Graph) {
	// Create generator network
	generatorGraph := tf.NewGraph()
	generatorScope := generatorGraph.Scope("generator")
	input := op.Placeholder(generatorGraph, tf.Float, op.PlaceholderShape(tf.MakeShape(16, 100)))
	output := generator(generatorScope, input)

	discriminatorGraph := tf.NewGraph()
	discriminatorScope := discriminatorGraph.Scope("discriminator")
	input = op.Placeholder(discriminatorGraph, tf.Float, op.PlaceholderShape(tf.MakeShape(16, 64, 64, 3)))
	output = discriminator(discriminatorScope, input)

	return generatorGraph, discriminatorGraph
}

func createLossFunctions() (*tf.Operation, *tf.Operation) {
	// Create generator loss function
	generatorGraph := tf.NewGraph()
	generatorScope := generatorGraph.Scope("generator")
	fakeImages := op.Placeholder(generatorGraph, tf.Float, op.PlaceholderShape(tf.MakeShape(16, 64, 64, 3)))
	discriminatorOutput := discriminator(generatorScope.SubScope("discriminator"), fakeImages)
	generatorLoss := op.Mean(generatorGraph, op.SigmoidCrossEntropyWithLogits(generatorGraph, op.OnesLike(generatorGraph, discriminatorOutput), discriminatorOutput, op.SigmoidCrossEntropyWithLogitsAttrs()))

	discriminatorGraph := tf.NewGraph()
	discriminatorScope := discriminatorGraph.Scope("discriminator")
	realImages := op.Placeholder(discriminatorGraph, tf.Float, op.PlaceholderShape(tf.MakeShape(16, 64, 64, 3)))
	fakeImages = op.Placeholder(discriminatorGraph, tf.Float, op.PlaceholderShape(tf.MakeShape(16, 64, 64, 3)))
	realDiscriminatorOutput := discriminator(discriminatorScope.SubScope("real"), realImages)
	fakeDiscriminatorOutput := discriminator(discriminatorScope.SubScope("fake"), fakeImages)
	realDiscriminatorLoss := op.Mean(discriminatorGraph, op.SigmoidCrossEntropyWithLogits(discriminatorGraph, op.OnesLike(discriminatorGraph, realDiscriminatorOutput), realDiscriminatorOutput, op.SigmoidCrossEntropyWithLogitsAttrs()))
	fakeDiscriminatorLoss := op.Mean(discriminatorGraph, op.SigmoidCrossEntropyWithLogits(discriminatorGraph, op.ZerosLike(discriminatorGraph, fakeDiscriminatorOutput), fakeDiscriminatorOutput, op.SigmoidCrossEntropyWithLogitsAttrs()))
	discriminatorLoss := op.Add(discriminatorGraph, realDiscriminatorLoss, fakeDiscriminatorLoss)

	return generatorLoss, discriminatorLoss
}

func createOptimizer() *tf.Operation {
	// Create optimizer
	graph := tf.NewGraph()
	optimizer := op.ApplyAdam(graph, op.Const(graph, 0.0002), op.Const(graph, 0.5), op.Const(graph, 0.999), op.Const(graph, 1e-8), 1, op.ApplyAdamUseLocking(false), op.ApplyAdamBeta1Power(0.9), op.ApplyAdamBeta2Power(0.999))

	return optimizer
}
