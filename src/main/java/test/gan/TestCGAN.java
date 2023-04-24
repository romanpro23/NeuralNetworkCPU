package test.gan;

import data.ImageCreator;
import data.loaders.TransformData;
import data.mnist.BatchMNIST;
import data.mnist.MNISTLoader3D;
import data.svhn.SVHNLoader3D;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.layer_3d.ConvolutionLayer;
import neural_network.layers.layer_3d.ConvolutionTransposeLayer;
import neural_network.layers.layer_3d.InstanceNormalizationLayer3D;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.reshape.FlattenLayer3D;
import neural_network.loss.FunctionLoss;
import neural_network.network.GAN.CycleGAN;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

import java.io.File;
import java.io.FileWriter;

public class TestCGAN {
    public static void main(String[] args) throws Exception {
        NeuralNetwork discriminatorA = new NeuralNetwork()
                .addInputLayer(32, 32, 3)
                .addLayer(new ConvolutionLayer(32, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new ConvolutionLayer(64, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new ConvolutionLayer(128, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new FlattenLayer3D())
                .addLayer(new DenseLayer(1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addActivationLayer(new FunctionActivation.Sigmoid())
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0002))
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .create();

        NeuralNetwork discriminatorB = new NeuralNetwork()
                .addInputLayer(28, 28, 1)
                .addLayer(new ConvolutionLayer(32, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new ConvolutionLayer(64, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new ConvolutionLayer(128, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new FlattenLayer3D())
                .addLayer(new DenseLayer(1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addActivationLayer(new FunctionActivation.Sigmoid())
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0002))
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .create();
//
        NeuralNetwork generatorA = new NeuralNetwork()
                .addInputLayer(28, 28, 1)
                .addLayer(new ConvolutionLayer(32, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new ConvolutionLayer(64, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new ConvolutionLayer(128, 4, 2, 2).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new ConvolutionTransposeLayer(64, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new ConvolutionTransposeLayer(32, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new ConvolutionTransposeLayer(3, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addActivationLayer(new FunctionActivation.Tanh())
                .setFunctionLoss(new FunctionLoss.MSE())
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0002))
                .create();

        NeuralNetwork generatorB = new NeuralNetwork()
                .addInputLayer(32, 32, 3)
                .addLayer(new ConvolutionLayer(32, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new ConvolutionLayer(64, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new ConvolutionLayer(128, 4, 1, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.LeakyReLU(0.2))
                .addLayer(new ConvolutionTransposeLayer(64, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new ConvolutionTransposeLayer(32, 4, 2, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addLayer(new InstanceNormalizationLayer3D())
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new ConvolutionLayer(1, 3, 1, 1).setInitializer(new Initializer.RandomNormal(0.02)))
                .addActivationLayer(new FunctionActivation.Tanh())
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0002))
                .setFunctionLoss(new FunctionLoss.MSE())
                .create();

        CycleGAN gan = new CycleGAN(generatorA, discriminatorA, generatorB, discriminatorB).setIdentityLoss(false);
        SVHNLoader3D loaderSVHN = new SVHNLoader3D(new TransformData.Tanh());
        MNISTLoader3D loaderMNIST = new MNISTLoader3D(BatchMNIST.MNIST,new TransformData.Tanh());

        discriminatorA.info();
        discriminatorB.info();

        generatorA.info();
        generatorB.info();

        for (int i = 0; i < 10000; i++) {
            long start = System.nanoTime();
//            if (i % 10 == 0) {
                NNTensor[] data = NNArrays.isTensor(loaderMNIST.getNextTestData(1).getInput());
                ImageCreator.drawImage(data[0], 28, 28, i / 10 + "_input", "D:/NetworkTest/CycleGAN", true);
                ImageCreator.drawColorImage((NNTensor) gan.query(data)[0], 32, 32, i / 10 + "_output", "D:/NetworkTest/CycleGAN", true);
                ImageCreator.drawImage((NNTensor) generatorB.query(generatorA.getOutputs())[0], 28, 28, i / 10 + "_recon", "D:/NetworkTest/CycleGAN", true);

                generatorA.save(new FileWriter(new File("D:/NetworkTest/CycleGAN/generator_svhn.txt")));
                discriminatorA.save(new FileWriter(new File("D:/NetworkTest/CycleGAN/discriminator_svhn.txt")));
                generatorB.save(new FileWriter(new File("D:/NetworkTest/CycleGAN/generator_mnist.txt")));
                discriminatorB.save(new FileWriter(new File("D:/NetworkTest/CycleGAN/discriminator_mnist.txt")));
//            }
            System.out.println(i + " - " + gan.train(loaderMNIST.getNextTrainData(64).getInput(), loaderSVHN.getNextTrainData(64).getInput()));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
