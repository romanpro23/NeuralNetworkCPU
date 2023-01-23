package test.decolorize;

import data.ImageCreator;
import data.imageNet.ImageNetLoader3D;
import data.loaders.TransformData;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.NeuralLayer;
import neural_network.layers.layer_3d.*;
import neural_network.loss.FunctionLoss;
import neural_network.network.GAN.Pix2PixGAN;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Scanner;

public class Pix2PixDecolorize {
    public static void main(String[] args) throws Exception {
//        NeuralNetwork generator = new NeuralNetwork()
//                .addInputLayer(64, 64, 1)
//                .addLayers(downsample(32, 4, false))
//                .addLayers(downsample(64, 4, true))
//                .addLayers(downsample(128, 4, true))
//                .addLayers(downsample(256, 4, true))
//                .addLayers(downsample(512, 4, true))
//                .addLayers(downsample(512, 4, true))
//                .addLayers(upsample(512, 4, true))
//                .addUConcatenateLayer(13)
//                .addLayers(upsample(256, 4, true))
//                .addUConcatenateLayer(10)
//                .addLayers(upsample(128, 4, true))
//                .addUConcatenateLayer(7)
//                .addLayers(upsample(64, 4, true))
//                .addUConcatenateLayer(4)
//                .addLayers(upsample(32, 4, true))
//                .addUConcatenateLayer(1)
//                .addLayer(new ConvolutionTransposeLayer(3, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
//                .addActivationLayer(new FunctionActivation.Tanh())
//                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0002))
//                .setFunctionLoss(new FunctionLoss.MAE())
//                .create();

        NeuralNetwork generator = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/Decolorize/Pix2Pix/u-net_generator.txt")))
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0002))
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();
        generator.info();

//        NeuralNetwork discriminator = new NeuralNetwork()
//                .addInputLayer(64, 64, 4)
//                .addLayers(downsample(32, 4, false))
//                .addLayers(downsample(64, 4, true))
//                .addLayers(downsample(128, 4, true))
//                .addLayer(new ConvolutionLayer(256, 4, 1, 1).setInitializer(new Initializer.RandomNormal(0.02)))
//                .addLayer(new BatchNormalizationLayer3D())
//                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
//                .addLayer(new ConvolutionLayer(1, 4, 1, 1).setInitializer(new Initializer.RandomNormal(0.02)))
//                .addActivationLayer(new FunctionActivation.Sigmoid())
//                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
//                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0001))
//                .create();

        NeuralNetwork discriminator = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/Decolorize/Pix2Pix/patch_discriminator.txt")))
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0002))
                .create();
        discriminator.info();

        Pix2PixGAN gan = new Pix2PixGAN(generator, discriminator);

        ImageNetLoader3D loader = new ImageNetLoader3D(64, new TransformData.Tanh());

        float lambda = 100;

        for (int i = 0; i < 100000; i++) {
            long start = System.nanoTime();
            if (i % 10 == 0) {
                NNTensor[] data = NNArrays.isTensor(loader.getNextTestData(1).getInput());
                NNTensor[] decolordata = ImageCreator.decolorize(data);
                ImageCreator.drawColorImage(data[0], 64, 64, i / 10 + "_color", "D:/NetworkTest/Decolorize/Pix2Pix", true);
                ImageCreator.drawImage(decolordata[0], 64, 64, i / 10 + "_black", "D:/NetworkTest/Decolorize/Pix2Pix", true);
                generator.save(new FileWriter(new File("D:/NetworkTest/Decolorize/Pix2Pix/u-net_generator.txt")));
                discriminator.save(new FileWriter(new File("D:/NetworkTest/Decolorize/Pix2Pix/patch_discriminator.txt")));

                NNTensor resultRandom = NNArrays.isTensor(gan.query(decolordata))[0];
                ImageCreator.drawColorImage(resultRandom, 64, 64, i / 10 + "_test", "D:/NetworkTest/Decolorize/Pix2Pix", true);
            }
            NNTensor[] train = NNArrays.isTensor(loader.getNextTrainData(64).getInput());
            System.out.println(i + " - " + gan.train(ImageCreator.decolorize(train), train, lambda));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }

    public static ArrayList<NeuralLayer> downsample(int countFilters, int sizeCore, boolean batchNorm) {
        ArrayList<NeuralLayer> layers = new ArrayList<>();

        Initializer initializer = new Initializer.RandomNormal(0.02);
        layers.add(new ConvolutionLayer(countFilters, sizeCore, 2, 1).setInitializer(initializer));
        if (batchNorm)
            layers.add(new BatchNormalizationLayer3D());
        layers.add(new ActivationLayer3D(new FunctionActivation.LeakyReLU(0.2)));

        return layers;
    }

    public static ArrayList<NeuralLayer> upsample(int countFilters, int sizeCore, boolean dropout) {
        ArrayList<NeuralLayer> layers = new ArrayList<>();

        Initializer initializer = new Initializer.RandomNormal(0.02);
        layers.add(new ConvolutionTransposeLayer(countFilters, sizeCore, 2, 1).setInitializer(initializer));
        layers.add(new BatchNormalizationLayer3D());
        if (dropout)
            layers.add(new DropoutLayer3D(0.5));
        layers.add(new ActivationLayer3D(new FunctionActivation.ReLU()));

        return layers;
    }
}
