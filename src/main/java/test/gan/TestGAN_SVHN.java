package test.gan;

import data.ImageCreator;
import data.loaders.TransformData;
import data.svhn.SVHNLoader3D;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.layer_1d.SNDenseLayer;
import neural_network.layers.layer_3d.*;
import neural_network.layers.reshape.FlattenLayer3D;
import neural_network.layers.reshape.ReshapeLayer3D;
import neural_network.loss.FunctionLoss;
import neural_network.network.GAN.GAN;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.File;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.Scanner;

public class TestGAN_SVHN {
    static String path = "D:/NetworkTest/svhn/GAN_SN";
    public static void main(String[] args) throws Exception {
        Optimizer optimizerD = new AdamOptimizer(0.5, 0.99, 0.0002);
        NeuralNetwork discriminator = new NeuralNetwork()
                .addInputLayer(32, 32, 3)
                .addLayer(new SNConvolutionLayer(32, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new SNConvolutionLayer(64, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new BatchNormalizationLayer3D(0.9))
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new SNConvolutionLayer(128, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new BatchNormalizationLayer3D(0.9))
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new FlattenLayer3D())
                .addLayer(new SNDenseLayer(1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addActivationLayer(new FunctionActivation.Sigmoid())
                .setOptimizer(optimizerD)
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .create();

        Optimizer optimizerG = new AdamOptimizer(0.5, 0.99, 0.0002);
        NeuralNetwork generator = new NeuralNetwork()
                .addInputLayer(100)
                .addLayer(new SNDenseLayer(2048).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new ReshapeLayer3D(4, 4, 128))
                .addLayer(new SNConvolutionTransposeLayer(64, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new BatchNormalizationLayer3D(0.9))
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new SNConvolutionTransposeLayer(32, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new BatchNormalizationLayer3D(0.9))
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new SNConvolutionTransposeLayer(3, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addActivationLayer(new FunctionActivation.Tanh())
                .setOptimizer(optimizerG)
                .create();

//        Optimizer optimizerD = new AdamOptimizer(0.5, 0.99, 0.0002);
//        NeuralNetwork discriminator = NeuralNetwork
//                .read(path + "/discriminator.txt")
//                .setOptimizer(optimizerD)
//                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
//                .create();
//        optimizerD.read(path + "/discriminator_optimizer.txt");
//
//        Optimizer optimizerG = new AdamOptimizer(0.5, 0.99, 0.0002);
//        NeuralNetwork generator = NeuralNetwork
//                .read(path + "/generator.txt")
//                .setOptimizer(optimizerG)
//                .create();
//        optimizerG.read(path + "/generator_optimizer.txt");

        GAN gan = new GAN(generator, discriminator).setInitializer(new Initializer.RandomNormal());
        SVHNLoader3D loader = new SVHNLoader3D(new TransformData.Tanh());
        Initializer initializer = new Initializer.RandomNormal();

        generator.info();
        discriminator.info();

        for (int i = 0; i < 1000; i++) {
            long start = System.nanoTime();
            if (i % 10 == 0) {
                NNTensor[] data = NNArrays.isTensor(loader.getNextTestData(1).getInput());
                ImageCreator.drawColorImage(data[0], 32, 32, i / 10 + "_input", path, true);

                generator.save(path + "/generator.txt");
                optimizerG.save(path + "/generator_optimizer.txt");
                discriminator.save(path + "/discriminator.txt");
                optimizerD.save(path + "/discriminator_optimizer.txt");

                NNVector[] random = new NNVector[1];
                random[0] = new NNVector(100);
                initializer.initialize(random[0]);
                NNTensor resultRandom = NNArrays.isTensor(gan.query(random))[0];
                ImageCreator.drawColorImage(resultRandom, 32, 32, i / 10 + "_test", path, true);
            }

            System.out.println(i + " - " + Arrays.toString(gan.train(loader.getNextTrainData(128).getInput())));
            System.out.println((System.nanoTime() - start) / 1000000);
        }

//        for (int i = 0; i < 100; i++) {
//            NNVector[] random = new NNVector[1];
//            random[0] = new NNVector(100);
//            initializer.initialize(random[0]);
//            NNTensor resultRandom = NNArrays.isTensor(gan.query(random))[0];
//            ImageCreator.drawColorImage(resultRandom, 32, 32, i + "_sample", "D:/NetworkTest/svhn/GAN", true);
//        }
    }
}
