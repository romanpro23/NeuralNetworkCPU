package test.gan;

import data.ImageCreator;
import data.loaders.TransformData;
import data.mnist.BatchMNIST;
import data.mnist.MNISTLoader1D;
import data.svhn.SVHNLoader1D;
import data.svhn.SVHNLoader3D;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.convolution_3d.BatchNormalizationLayer3D;
import neural_network.layers.convolution_3d.ConvolutionLayer;
import neural_network.layers.convolution_3d.ConvolutionTransposeLayer;
import neural_network.layers.dense.BatchNormalizationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.layers.dense.DropoutLayer;
import neural_network.layers.reshape.Flatten3DLayer;
import neural_network.layers.reshape.Reshape3DLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.GAN.GAN;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class TestGAN_SVHN {
    public static void main(String[] args) throws Exception {
        NeuralNetwork discriminator = new NeuralNetwork()
                .addInputLayer(32, 32, 3)
                .addLayer(new ConvolutionLayer(32, 4, 2, 1).setInitializer(new Initializer.RandomUniform(0.02)))
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new ConvolutionLayer(64, 4, 2, 1).setInitializer(new Initializer.RandomUniform(0.02)))
                .addLayer(new BatchNormalizationLayer3D(0.1))
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new ConvolutionLayer(128, 4, 2, 1).setInitializer(new Initializer.RandomUniform(0.02)))
                .addLayer(new BatchNormalizationLayer3D(0.1))
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new Flatten3DLayer())
                .addLayer(new DenseLayer(1).setInitializer(new Initializer.RandomUniform(0.02)))
                .addActivationLayer(new FunctionActivation.Sigmoid())
                .setOptimizer(new AdamOptimizer(0.5, 0.99, 0.0002))
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .create();

        NeuralNetwork generator = new NeuralNetwork()
                .addInputLayer(100)
                .addLayer(new DenseLayer(2048).setInitializer(new Initializer.RandomUniform(0.02)))
                .addLayer(new Reshape3DLayer(4, 4, 128))
                .addLayer(new ConvolutionTransposeLayer(64, 4, 2, 1).setInitializer(new Initializer.RandomUniform(0.02)))
                .addLayer(new BatchNormalizationLayer3D(0.1))
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new ConvolutionTransposeLayer(32, 4, 2, 1).setInitializer(new Initializer.RandomUniform(0.02)))
                .addLayer(new BatchNormalizationLayer3D(0.1))
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new ConvolutionTransposeLayer(3, 4, 2, 1).setInitializer(new Initializer.RandomUniform(0.02)))
                .addActivationLayer(new FunctionActivation.Tanh())
                .setOptimizer(new AdamOptimizer(0.5, 0.99, 0.0002))
                .create();

//        NeuralNetwork discriminator = NeuralNetwork
//                .read(new Scanner(new File("D:/NetworkTest/svhn/GAN/discriminator_conv_svhn.txt")))
//                .setOptimizer(new AdamOptimizer(0.5, 0.99, 0.0002))
//                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
//                .create();
//
//        NeuralNetwork generator = NeuralNetwork
//                .read(new Scanner(new File("D:/NetworkTest/svhn/GAN/generator_conv_svhn.txt")))
//                .setOptimizer(new AdamOptimizer(0.5, 0.99, 0.0002))
//                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
//                .create();


        GAN gan = new GAN(generator, discriminator);
        gan.setInitializer(new Initializer.RandomUniform(2));
        SVHNLoader3D loader = new SVHNLoader3D(new TransformData.Tanh());
        Initializer initializer = new Initializer.RandomUniform(2);

        generator.info();
        discriminator.info();

        for (int i = 0; i < 100000; i++) {
            long start = System.nanoTime();
            if (i % 25 == 0) {
                NNTensor[] data = NNArrays.isTensor(loader.getNextTestData(1).getInput());
                ImageCreator.drawColorImage(data[0], 32, 32, i / 25 + "_input", "D:/NetworkTest/svhn/GAN", true);
                generator.save(new FileWriter(new File("D:/NetworkTest/svhn/GAN/generator_conv_svhn.txt")));
                discriminator.save(new FileWriter(new File("D:/NetworkTest/svhn/GAN/discriminator_conv_svhn.txt")));

                NNVector[] random = new NNVector[1];
                random[0] = new NNVector(100);
                initializer.initialize(random[0]);
                NNTensor resultRandom = NNArrays.isTensor(gan.query(random))[0];
                ImageCreator.drawColorImage(resultRandom, 32, 32, i / 25 + "_test", "D:/NetworkTest/svhn/GAN", true);
            }
            if (i % 200 == 0) {
                for (int j = 0; j < 10; j++) {
                    NNVector[] random = new NNVector[1];
                    random[0] = new NNVector(100);
                    initializer.initialize(random[0]);
                    NNTensor resultRandom = NNArrays.isTensor(gan.query(random))[0];
                    ImageCreator.drawColorImage(resultRandom, 32, 32, i / 200 + "_random_" + j, "D:/NetworkTest/svhn/GAN", true);
                }
            }
            System.out.println(i + " - " + gan.train(loader.getNextTrainData(64).getInput()));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
