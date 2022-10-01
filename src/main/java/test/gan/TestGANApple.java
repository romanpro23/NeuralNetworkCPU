package test.gan;

import data.ImageCreator;
import data.quick_draw.QuickDrawLoader3D;
import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.convolution_3d.*;
import neural_network.layers.reshape.Flatten3DLayer;
import neural_network.layers.reshape.Reshape3DLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.GAN.GAN;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdaDeltaOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.FileWriter;

public class TestGANApple {
    public static void main(String[] args) throws Exception {
        NeuralNetwork discriminator = new NeuralNetwork()
                .addInputLayer(28, 28, 1)
                .addLayer(new ConvolutionLayer(16, 5, 2, 2))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new DropoutLayer3D(0.4))
                .addLayer(new ConvolutionLayer(32, 5, 2, 2))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new DropoutLayer3D(0.4))
                .addLayer(new ConvolutionLayer(64, 5, 1, 2))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new DropoutLayer3D(0.4))
                .addLayer(new ConvolutionLayer(128, 5, 1, 2))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new DropoutLayer3D(0.4))
                .addLayer(new Flatten3DLayer())
                .addDenseLayer(1, new FunctionActivation.Sigmoid())
                .setOptimizer(new AdaDeltaOptimizer(0.0008, 0.00000006))
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .create();

        NeuralNetwork generator = new NeuralNetwork()
                .addInputLayer(32)
                .addDenseLayer(3136)
                .addBatchNormalizationLayer(0.9)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDropoutLayer(0.25)
                .addLayer(new Reshape3DLayer(7, 7, 64))
                .addLayer(new UpSamplingLayer(2))
                .addLayer(new ConvolutionTransposeLayer(32, 5, 1, 2)
                        .setInitializer(new Initializer.RandomUniform()))
                .addLayer(new BatchNormalizationLayer3D(0.9))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new UpSamplingLayer(2))
                .addLayer(new ConvolutionTransposeLayer(16, 5, 1, 2))
                .addLayer(new BatchNormalizationLayer3D(0.9))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new ConvolutionTransposeLayer(8, 5, 1, 2)
                    .setInitializer(new Initializer.RandomUniform()))
                .addLayer(new BatchNormalizationLayer3D(0.9))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new ConvolutionLayer(1, 5, 1, 2))
                .addLayer(new ActivationLayer3D(new FunctionActivation.Sigmoid()))
                .setOptimizer(new AdaDeltaOptimizer(0.0004, 0.00000003))
                .create();

//        NeuralNetwork generator = NeuralNetwork
//                .read(new Scanner(new File("D:/NetworkTest/CGAN/generator_apples.txt")))
//                .setOptimizer(new AdaDeltaOptimizer(0.0004, 0.00000003))
//                .create();
//
//        NeuralNetwork discriminator = NeuralNetwork
//                .read(new Scanner(new File("D:/NetworkTest/CGAN/discriminator_apples.txt")))
//                .setOptimizer(new AdaDeltaOptimizer(0.0008, 0.00000006))
//                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
//                .create();

        generator.info();
        discriminator.info();

        GAN gan = new GAN(generator, discriminator);

        QuickDrawLoader3D loader = new QuickDrawLoader3D(10000, 1000, "apple");
        Initializer init = new Initializer.RandomNormal();

        for (int i = 0; i < 100000; i++) {
            long start = System.nanoTime();
            if (i % 25 == 0) {
                NNTensor[] data = NNArrays.isTensor(loader.getNextTestData(1).getInput());
                ImageCreator.drawImage(data[0], 28, 28, i / 25 + "_input", "D:/NetworkTest/GAN");
                generator.save(new FileWriter("D:/NetworkTest/CGAN/generator_apples.txt"));
                discriminator.save(new FileWriter("D:/NetworkTest/CGAN/discriminator_apples.txt"));

                NNVector[] random = new NNVector[1];
                random[0] = new NNVector(32);
                init.initialize(random[0]);
                NNTensor resultRandom = NNArrays.toTensor(gan.query(random), 28, 28, 1)[0];
                ImageCreator.drawImage(resultRandom, 28, 28, i / 25 + "_test", "D:/NetworkTest/CGAN");
            }
            if (i % 200 == 0) {
                for (int j = 0; j < 10; j++) {
                    NNVector[] random = new NNVector[1];
                    random[0] = new NNVector(32);
                    init.initialize(random[0]);
                    NNTensor resultRandom = NNArrays.toTensor(gan.query(random), 28, 28, 1)[0];
                    ImageCreator.drawImage(resultRandom, 28, 28, i / 200 + "_random_" + j, "D:/NetworkTest/CGAN");
                }
            }
            System.out.println(i + " - " + gan.train(loader.getNextTrainData(64).getInput()));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }
}
