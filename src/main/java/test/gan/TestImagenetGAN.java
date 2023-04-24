package test.gan;

import data.ImageCreator;
import data.imageNet.ImageNet250Loader3D;
import data.loaders.TransformData;
import data.network_train.NNData3D;
import neural_network.initialization.Initializer;
import neural_network.loss.FunctionLoss;
import neural_network.network.GAN.GAN;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.FileWriter;

public class TestImagenetGAN {
    static String path = "D:/NetworkTest/ImagenetGAN/GAN";

    public static void main(String[] args) throws Exception {
        Optimizer optimizerD = new AdamOptimizer(0.5, 0.99, 0.0002);
        NeuralNetwork discriminator = NeuralNetwork
                .read(path + "/discriminator.txt")
                .setOptimizer(optimizerD)
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .create();
        optimizerD.read(path + "/optimizer_discriminator.txt");

        Optimizer optimizerG = new AdamOptimizer(0.5, 0.99, 0.0002);
        NeuralNetwork generator = NeuralNetwork
                .read(path + "/generator.txt")
                .setOptimizer(optimizerG)
                .create();
        optimizerG.read(path + "/optimizer_generator.txt");

        GAN gan = new GAN(generator, discriminator);
        gan.setInitializer(new Initializer.RandomNormal());
        ImageNet250Loader3D loader = new ImageNet250Loader3D(new TransformData.Tanh()).useCrop().useReverse();
        Initializer initializer = new Initializer.RandomNormal();

        generator.info();
        discriminator.info();

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
