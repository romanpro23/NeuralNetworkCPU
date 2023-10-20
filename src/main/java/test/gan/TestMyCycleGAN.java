package test.gan;

import data.ImageCreator;
import data.image2image.AppleToOrangeLoader3D;
import data.loaders.TransformData;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.layer_3d.ActivationLayer3D;
import neural_network.layers.layer_3d.ConvolutionLayer;
import neural_network.layers.layer_3d.ConvolutionTransposeLayer;
import neural_network.layers.layer_3d.InstanceNormalizationLayer3D;
import neural_network.layers.layer_3d.residual.ResidualUnit;
import neural_network.layers.layer_3d.residual.ResidualBlock;
import neural_network.loss.FunctionLoss;
import neural_network.network.GAN.CycleGAN;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class TestMyCycleGAN {
    public static void main(String[] args) throws Exception {
        /*NeuralNetwork discriminatorA = getDiscriminator();

        NeuralNetwork discriminatorB = getDiscriminator();

        NeuralNetwork generatorA = getGenerator();

        NeuralNetwork generatorB = getGenerator();*/

        NeuralNetwork discriminatorA = NeuralNetwork
                .read(new Scanner(new File("C:/NetworkTest/CycleGAN/patch_discriminator_orange.txt")))
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0001))
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();

        NeuralNetwork discriminatorB = NeuralNetwork
                .read(new Scanner(new File("C:/NetworkTest/CycleGAN/patch_discriminator_apple.txt")))
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0001))
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();

        NeuralNetwork generatorA = NeuralNetwork
                .read(new Scanner(new File("C:/NetworkTest/CycleGAN/res_generator_orange.txt")))
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0001))
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();

        NeuralNetwork generatorB = NeuralNetwork
                .read(new Scanner(new File("C:/NetworkTest/CycleGAN/res_generator_apple.txt")))
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0001))
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();

        System.out.println(generatorA.getLayers().size());

        CycleGAN gan = new CycleGAN(generatorA, discriminatorA, generatorB, discriminatorB);
        AppleToOrangeLoader3D loader = new AppleToOrangeLoader3D(new TransformData.Tanh());

        discriminatorA.info();
        generatorB.info();

        int n = 1;
        for (int i = 0; i < 10000; i++) {
            long start = System.nanoTime();
            if (i % n == 0) {
                NNTensor[] data = NNArrays.isTensor(loader.getNextTestAData(1));
                ImageCreator.drawColorImage(data[0], 64, 64, i / n + "_apple", "C:/NetworkTest/CycleGAN", true);
                ImageCreator.drawColorImage((NNTensor) gan.query(data)[0], 64, 64, i / n + "_apple_orange", "C:/NetworkTest/CycleGAN", true);
                ImageCreator.drawColorImage((NNTensor) generatorB.query(generatorA.getOutputs())[0], 64, 64, i / n + "_apple_recon", "C:/NetworkTest/CycleGAN", true);

                data = NNArrays.isTensor(loader.getNextTestBData(1));
                ImageCreator.drawColorImage(data[0], 64, 64, i / n + "_orange", "C:/NetworkTest/CycleGAN", true);
                ImageCreator.drawColorImage((NNTensor) generatorB.query(data)[0], 64, 64, i / n + "_orange_apple", "C:/NetworkTest/CycleGAN", true);
                ImageCreator.drawColorImage((NNTensor) generatorA.query(generatorB.getOutputs())[0], 64, 64, i / n + "_orange_recon", "C:/NetworkTest/CycleGAN", true);

                generatorA.save(new FileWriter(new File("C:/NetworkTest/CycleGAN/res_generator_orange.txt")));
                discriminatorA.save(new FileWriter(new File("C:/NetworkTest/CycleGAN/patch_discriminator_orange.txt")));
                generatorB.save(new FileWriter(new File("C:/NetworkTest/CycleGAN/res_generator_apple.txt")));
                discriminatorB.save(new FileWriter(new File("C:/NetworkTest/CycleGAN/patch_discriminator_apple.txt")));
            }
            NNTensor[] apples = loader.getNextTrainAData(16);
            NNTensor[] oranges = loader.getNextTrainBData(16);
            if(apples.length != oranges.length){
                apples = loader.getNextTrainAData(oranges.length);
            }
            System.out.println(i + " - " + gan.train(apples,
                    oranges,
                    10,
                    5));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }

    public static NeuralNetwork getDiscriminator() {
        NeuralNetwork discriminator = new NeuralNetwork()
                .addInputLayer(64, 64, 3)
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
                .addInputLayer(64, 64, 3)
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

    private static ResidualBlock getResModule(){
        return new ResidualBlock()
                .addResidualUnit(new ResidualUnit())
                .addResidualUnit(new ResidualUnit()
                        .addLayer(new ConvolutionLayer(128, 3, 1, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                        .addLayer(new InstanceNormalizationLayer3D())
                        .addLayer( new ActivationLayer3D(new FunctionActivation.ReLU()))
                        .addLayer(new ConvolutionLayer(128, 3, 1, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                        .addLayer(new InstanceNormalizationLayer3D())
                );
    }
}
