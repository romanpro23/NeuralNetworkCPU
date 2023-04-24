package test.gan;

import data.ImageCreator;
import data.imageNet.ImageNet250Loader3D;
import data.loaders.TransformData;
import data.network_train.NNData3D;
import data.svhn.SVHNLoader3D;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.LayersBlock;
import neural_network.layers.layer_1d.ActivationLayer;
import neural_network.layers.layer_1d.BatchNormalizationLayer;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.layer_3d.BatchNormalizationLayer3D;
import neural_network.layers.layer_3d.ConvolutionLayer;
import neural_network.layers.layer_3d.ConvolutionTransposeLayer;
import neural_network.layers.reshape.EmbeddingLayer3D;
import neural_network.layers.reshape.FlattenLayer3D;
import neural_network.layers.reshape.ReshapeLayer3D;
import neural_network.loss.FunctionLoss;
import neural_network.network.GAN.ClassifierDiscriminator;
import neural_network.network.GAN.ConditionalGenerator;
import neural_network.network.GAN.InfoGAN;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.File;
import java.io.FileWriter;

public class TestInfoGAN {
    public static void main(String[] args) throws Exception {
        ClassifierDiscriminator discriminator = new ClassifierDiscriminator()
                .setDiscriminator(
                        new NeuralNetwork()
                                .addInputLayer(64, 64, 3)
                                .addLayer(new ConvolutionLayer(32, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                                .addLayer(new ConvolutionLayer(64, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                                .addLayer(new ConvolutionLayer(128, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                                .addLayer(new ConvolutionLayer(256, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                                .addLayer(new FlattenLayer3D())

                )
                .setDiscriminatorLayers(
                        new LayersBlock()
                                .addLayer(new DenseLayer(1024))
                                .addLayer(new BatchNormalizationLayer(0.9))
                                .addLayer(new ActivationLayer(new FunctionActivation.LeakyReLU(0.2)))
                                .addLayer(new DenseLayer(1))
                                .addLayer(new ActivationLayer(new FunctionActivation.Sigmoid()))
                )
                .setClassifierLayers(
                        new LayersBlock()
                                .addLayer(new DenseLayer(1024))
                                .addLayer(new BatchNormalizationLayer(0.9))
                                .addLayer(new ActivationLayer(new FunctionActivation.LeakyReLU(0.2)))
                                .addLayer(new DenseLayer(250))
                                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()))
                )
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0002))
                .setGANFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .setClassifierFunctionLoss(new FunctionLoss.CrossEntropy())
                .create();

        ConditionalGenerator generator = new ConditionalGenerator()
                .setNoiseLayers(
                        new LayersBlock().addInputLayer(100)
                                .addLayer(new DenseLayer(4 * 4 * 128))
                                .addLayer(new BatchNormalizationLayer(0.9))
                                .addLayer(new ActivationLayer(new FunctionActivation.ReLU()))
                                .addLayer(new ReshapeLayer3D(4, 4, 128))
                )
                .setLabelLayers(
                        new LayersBlock().addInputLayer(250)
                                .addLayer(new EmbeddingLayer3D(4, 4, 16))
                )
                .setGenerator(
                        new NeuralNetwork()
                                .addInputLayer(4, 4, 128+16)
                                .addLayer(new ConvolutionTransposeLayer(64, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addActivationLayer(new FunctionActivation.ReLU())
                                .addLayer(new ConvolutionTransposeLayer(48, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addActivationLayer(new FunctionActivation.ReLU())
                                .addLayer(new ConvolutionTransposeLayer(32, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addActivationLayer(new FunctionActivation.ReLU())
                                .addLayer(new ConvolutionTransposeLayer(3, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                                .addActivationLayer(new FunctionActivation.Tanh())
                )
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0002))
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
//                .create();


        InfoGAN gan = new InfoGAN(generator, discriminator);
        gan.setInitializer(new Initializer.RandomNormal());
        ImageNet250Loader3D loader = new ImageNet250Loader3D(new TransformData.Tanh()).useCrop().useReverse();
        Initializer initializer = new Initializer.RandomNormal();

        generator.info();
        discriminator.info();

        for (int i = 0; i < 1000000; i++) {
            long start = System.nanoTime();
            if (i % 10 == 0) {
                NNTensor[] data = NNArrays.isTensor(loader.getNextTestData(1).getInput());
                ImageCreator.drawColorImage(data[0], 64, 64, i / 10 + "_input", "D:/NetworkTest/ImagenetGAN/InfoGAN", true);
                generator.save(new FileWriter("D:/NetworkTest/ImagenetGAN/InfoGAN/generator_conv.txt"));
                discriminator.save(new FileWriter(new File("D:/NetworkTest/ImagenetGAN/InfoGAN/discriminator_conv.txt")));

                NNVector[] random = new NNVector[1];
                random[0] = new NNVector(100);
                initializer.initialize(random[0]);
                NNVector[] label = new NNVector[1];
                label[0] = new NNVector(250);
                label[0].set((int)(Math.random() * 250), 1);
                NNTensor resultRandom = NNArrays.isTensor(gan.query(random, label))[0];
                ImageCreator.drawColorImage(resultRandom, 32, 32, i / 10 + "_test_" + label[0].indexMaxElement(), "D:/NetworkTest/ImagenetGAN/InfoGAN", true);
            }
            NNData3D train = loader.getNextTrainData(128);
            System.out.println(i + " - " + gan.train(train.getInput(), train.getOutput()));
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
