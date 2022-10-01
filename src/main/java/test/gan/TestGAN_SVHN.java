package test.gan;

import data.ImageCreator;
import data.loaders.TransformData;
import data.mnist.BatchMNIST;
import data.mnist.MNISTLoader1D;
import data.svhn.SVHNLoader1D;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.convolution_3d.BatchNormalizationLayer3D;
import neural_network.layers.dense.BatchNormalizationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.layers.dense.DropoutLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.GAN.GAN;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNVector;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class TestGAN_SVHN {
    public static void main(String[] args) throws Exception {
        NeuralNetwork discriminator = new NeuralNetwork()
                .addInputLayer(3072)
                .addLayer(new DenseLayer(1536))
                .addLayer(new BatchNormalizationLayer())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DropoutLayer(0.3))
                .addLayer(new DenseLayer(718))
                .addLayer(new BatchNormalizationLayer())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DropoutLayer(0.3))
                .addLayer(new DenseLayer(256))
                .addLayer(new BatchNormalizationLayer())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DropoutLayer(0.3))
                .addLayer(new DenseLayer(1))
                .addActivationLayer(new FunctionActivation.Sigmoid())
                .setOptimizer(new AdamOptimizer(0.0001))
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .create();


        NeuralNetwork generator = new NeuralNetwork()
                .addInputLayer(100)
                .addLayer(new DenseLayer(256))
                .addLayer(new BatchNormalizationLayer())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DenseLayer(1024))
                .addLayer(new BatchNormalizationLayer())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DenseLayer(3072))
                .addLayer(new BatchNormalizationLayer())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(3072, new FunctionActivation.Sigmoid())
                .setOptimizer(new AdamOptimizer(0.0001))
                .create();

        GAN gan = new GAN(generator, discriminator);
        SVHNLoader1D loader = new SVHNLoader1D();
        Initializer initializer = new Initializer.RandomNormal();

        generator.info();
        discriminator.info();

        for (int i = 0; i < 100000; i++) {
            long start = System.nanoTime();
            if (i % 50 == 0) {
                NNVector[] data = NNArrays.isVector(loader.getNextTestData(1).getInput());
                ImageCreator.drawColorImage(data[0], 32, 32, i / 50 + "_input", "D:/NetworkTest/svhn/GAN");
                generator.save(new FileWriter(new File("D:/NetworkTest/svhn/GAN/generator_svhn.txt")));
                discriminator.save(new FileWriter(new File("D:/NetworkTest/svhn/GAN/discriminator_svhn.txt")));

                NNVector[] random = new NNVector[1];
                random[0] = new NNVector(100);
                initializer.initialize(random[0]);
                NNVector resultRandom = NNArrays.toVector(gan.query(random))[0];
                ImageCreator.drawColorImage(resultRandom, 32, 32, i / 50 + "_test", "D:/NetworkTest/svhn/GAN");
            }
            if (i % 500 == 0) {
                for (int j = 0; j < 10; j++) {
                    NNVector[] random = new NNVector[1];
                    random[0] = new NNVector(100);
                    initializer.initialize(random[0]);
                    NNVector resultRandom = NNArrays.toVector(gan.query(random))[0];
                    ImageCreator.drawColorImage(resultRandom, 32, 32, i / 500 + "_random_" + j, "D:/NetworkTest/svhn/GAN");
                }
            }
            System.out.println(i + " - " + gan.train(loader.getNextTrainData(64).getInput()));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
