package test.gan;

import data.ImageCreator;
import data.imageNet.ImageNet250Loader3D;
import data.loaders.TransformData;
import data.network_train.NNData3D;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.LayersBlock;
import neural_network.layers.layer_1d.ActivationLayer;
import neural_network.layers.layer_1d.BatchNormalizationLayer;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.layer_1d.SNDenseLayer;
import neural_network.layers.layer_3d.*;
import neural_network.layers.reshape.EmbeddingLayer3D;
import neural_network.layers.reshape.FlattenLayer3D;
import neural_network.layers.reshape.ReshapeLayer3D;
import neural_network.loss.FunctionLoss;
import neural_network.network.GAN.ClassifierDiscriminator;
import neural_network.network.GAN.ConditionalGenerator;
import neural_network.network.GAN.GAN;
import neural_network.network.GAN.InfoGAN;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.File;
import java.io.FileWriter;

public class TestGANImagenet {
    static String path = "D:/NetworkTest/ImagenetGAN/GAN";

    public static void main(String[] args) throws Exception {
        Optimizer optimizerD = new AdamOptimizer(0.5, 0.99, 0.0002);
        NeuralNetwork discriminator = new NeuralNetwork()
                .addInputLayer(64, 64, 3)
                .addLayer(new SNConvolutionLayer(32, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new SNConvolutionLayer(64, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new BatchNormalizationLayer3D(0.9))
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new SNConvolutionLayer(128, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new BatchNormalizationLayer3D(0.9))
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new SNConvolutionLayer(256, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new BatchNormalizationLayer3D(0.9))
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new FlattenLayer3D())
                .addLayer(new SNDenseLayer(1))
                .addLayer(new ActivationLayer(new FunctionActivation.Sigmoid()))
                .setOptimizer(optimizerD)
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .create();

        Optimizer optimizerG = new AdamOptimizer(0.5, 0.99, 0.0002);
        NeuralNetwork generator = new NeuralNetwork()
                .addInputLayer(100)
                .addLayer(new SNDenseLayer(4 * 4 * 256))
                .addLayer(new BatchNormalizationLayer(0.9))
                .addLayer(new ActivationLayer(new FunctionActivation.ReLU()))
                .addLayer(new ReshapeLayer3D(4, 4, 256))
                .addLayer(new SNConvolutionTransposeLayer(128, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new BatchNormalizationLayer3D(0.9))
                .addActivationLayer(new FunctionActivation.ReLU())
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
//        optimizerD.read(path + "/optimizer_discriminator.txt");
//
//        Optimizer optimizerG = new AdamOptimizer(0.5, 0.99, 0.0002);
//        NeuralNetwork generator = NeuralNetwork
//                .read(path + "/generator.txt")
//                .setOptimizer(optimizerG)
//                .create();
//        optimizerG.read(path + "/optimizer_generator.txt");

        GAN gan = new GAN(generator, discriminator);
        gan.setInitializer(new Initializer.RandomNormal());
        ImageNet250Loader3D loader = new ImageNet250Loader3D(new TransformData.Tanh()).useCrop().useReverse();
        Initializer initializer = new Initializer.RandomNormal();

        generator.info();
        discriminator.info();

        for (int i = 0; i < 10000; i++) {
            long start = System.nanoTime();
            if (i % 10 == 0) {
                NNTensor[] data = NNArrays.isTensor(loader.getNextTestData(1).getInput());
                ImageCreator.drawColorImage(data[0], 64, 64, i / 10 + "_input", path, true);
                generator.save(new FileWriter(path + "/generator_sn.txt"));
                optimizerG.save(path + "/optimizer_generator_sn.txt");
                optimizerD.save(path + "/optimizer_discriminator_sn.txt");
                discriminator.save(new FileWriter(path + "/discriminator_sn.txt"));

                NNVector[] random = new NNVector[1];
                random[0] = new NNVector(100);
                initializer.initialize(random[0]);
                NNTensor resultRandom = NNArrays.isTensor(gan.query(random))[0];
                ImageCreator.drawColorImage(resultRandom, 64, 64, i / 10 + "_test", path, true);
            }
            NNData3D train = loader.getNextTrainData(128);
            System.out.println(i + " - " + gan.train(train.getInput()));
            System.out.println((System.nanoTime() - start) / 1000000);
        }

        int size = 100;
        NNVector[] random = new NNVector[size];
        for (int i = 0; i < size; i++) {
            random[i] = new NNVector(100);
            initializer.initialize(random[i]);
        }
        NNTensor[] resultRandom = NNArrays.isTensor(gan.query(random));

        for (int i = 0; i < size; i++) {
            ImageCreator.drawColorImage(resultRandom[i], 64, 64, "sample_" + i, path, true);
        }

        NNTensor[] real = NNArrays.isTensor(loader.getNextTestData(size).getInput());

        for (int i = 0; i < size; i++) {
            ImageCreator.drawColorImage(real[i], 64, 64, "real_" + i, path, true);
        }
    }
}
