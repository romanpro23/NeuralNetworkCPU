package test.gan;

import data.ImageCreator;
import data.image2image.FishOldToNewLoader3D;
import data.loaders.TransformData;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.convolution_3d.ActivationLayer3D;
import neural_network.layers.convolution_3d.ConvolutionLayer;
import neural_network.layers.convolution_3d.ConvolutionTransposeLayer;
import neural_network.layers.convolution_3d.InstanceNormalizationLayer3D;
import neural_network.layers.convolution_3d.residual.ResidualBlock;
import neural_network.layers.convolution_3d.residual.ResidualModule;
import neural_network.loss.FunctionLoss;
import neural_network.network.GAN.CycleGAN;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class TestCycleGANFish {
    public static void main(String[] args) throws Exception {
//        NeuralNetwork discriminatorA = getDiscriminator();
//
//        NeuralNetwork discriminatorB = getDiscriminator();
//
//        NeuralNetwork generatorA = getGenerator();
//
//        NeuralNetwork generatorB = getGenerator();

        NeuralNetwork discriminatorA = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/Fish/patch_discriminator_new.txt")))
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0001))
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();

        NeuralNetwork discriminatorB = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/Fish/patch_discriminator_old.txt")))
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0001))
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();

        NeuralNetwork generatorA = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/Fish/res_generator_new.txt")))
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0002))
                .setFunctionLoss(new FunctionLoss.MAE())
                .create();

        NeuralNetwork generatorB = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/Fish/res_generator_old.txt")))
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0002))
                .setFunctionLoss(new FunctionLoss.MAE())
                .create();

        System.out.println(generatorA.getLayers().size());

        CycleGAN gan = new CycleGAN(generatorA, discriminatorA, generatorB, discriminatorB);
        FishOldToNewLoader3D loader = new FishOldToNewLoader3D(128, new TransformData.Tanh());

        discriminatorA.info();
        generatorB.info();

        int n = 1;
        for (int i = 0; i < 10000; i++) {
            long start = System.nanoTime();
            int size = 128;
            if (i % n == 0) {
                NNTensor[] data = NNArrays.isTensor(loader.getNextTrainAData(1));
                ImageCreator.drawColorImage(data[0], size, size, i / n + "_old", "D:/NetworkTest/Fish", true);
                ImageCreator.drawColorImage((NNTensor) gan.query(data)[0], size, size, i / n + "_old_new", "D:/NetworkTest/Fish", true);
                ImageCreator.drawColorImage((NNTensor) generatorB.query(generatorA.getOutputs())[0], size, size, i / n + "_old_recon", "D:/NetworkTest/Fish", true);

                data = NNArrays.isTensor(loader.getNextTrainBData(1));
                ImageCreator.drawColorImage(data[0], size, size, i / n + "_new", "D:/NetworkTest/Fish", true);
                ImageCreator.drawColorImage((NNTensor) generatorB.query(data)[0], size, size, i / n + "_new_old", "D:/NetworkTest/Fish", true);
                ImageCreator.drawColorImage((NNTensor) generatorA.query(generatorB.getOutputs())[0], size, size, i / n + "_new_recon", "D:/NetworkTest/Fish", true);

                generatorA.save(new FileWriter(new File("D:/NetworkTest/Fish/res_generator_new.txt")));
                discriminatorA.save(new FileWriter(new File("D:/NetworkTest/Fish/patch_discriminator_new.txt")));
                generatorB.save(new FileWriter(new File("D:/NetworkTest/Fish/res_generator_old.txt")));
                discriminatorB.save(new FileWriter(new File("D:/NetworkTest/Fish/patch_discriminator_old.txt")));
            }
            System.out.println(i + " - " + gan.train(
                    loader.getNextTrainAData(4),
                    loader.getNextTrainBData(4),
                    10,
                    5));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }

    public static NeuralNetwork getDiscriminator() {
        NeuralNetwork discriminator = new NeuralNetwork()
                .addInputLayer(128, 128, 3)
                .addLayer(new ConvolutionLayer(32, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new ConvolutionLayer(64, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new ConvolutionLayer(128, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new ConvolutionLayer(256, 4, 1, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new ConvolutionLayer(1, 4, 1, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addActivationLayer(new FunctionActivation.Sigmoid())
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0001))
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();

        return discriminator;
    }

    public static NeuralNetwork getGenerator() {
        NeuralNetwork generator = new NeuralNetwork()
                .addInputLayer(128, 128, 3)
                .addLayer(new ConvolutionLayer(32, 7, 1, 3).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new ConvolutionLayer(64, 3, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new ConvolutionLayer(128, 3, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(getResModule())
                .addLayer(getResModule())
                .addLayer(getResModule())
                .addLayer(getResModule())
                .addLayer(getResModule())
                .addLayer(getResModule())
                .addLayer(new ConvolutionTransposeLayer(64, 3, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new ConvolutionTransposeLayer(32, 3, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new ConvolutionLayer(3, 7, 1, 3).setInitializer(new Initializer.RandomNormal(0.02)))
                .addActivationLayer(new FunctionActivation.Tanh())
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0002))
                .setFunctionLoss(new FunctionLoss.MAE())
                .create();

        return generator;
    }

    private static ResidualModule getResModule(){
        return new ResidualModule()
                .addResidualBlock(new ResidualBlock())
                .addResidualBlock(new ResidualBlock()
                        .addLayer(new ConvolutionLayer(128, 3, 1, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                        .addLayer(new InstanceNormalizationLayer3D())
                        .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                        .addLayer(new ConvolutionLayer(128, 3, 1, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                        .addLayer(new InstanceNormalizationLayer3D())
                );
    }
}
