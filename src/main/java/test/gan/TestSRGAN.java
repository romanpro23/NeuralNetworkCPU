package test.gan;

import data.ImageCreator;
import data.imageNet.ScaleImageNetLoader3D;
import data.network_train.ImgNNData3D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.layer_3d.ActivationLayer3D;
import neural_network.layers.layer_3d.BatchNormalizationLayer3D;
import neural_network.layers.layer_3d.ConvolutionLayer;
import neural_network.layers.layer_3d.ParametricReLULayer3D;
import neural_network.layers.layer_3d.residual.ResidualUnit;
import neural_network.layers.layer_3d.residual.ResidualBlock;
import neural_network.layers.reshape.Flatten3DLayer;
import neural_network.layers.layer_3d.PixelShufflerLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.GAN.SRGAN;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNTensor;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class TestSRGAN {
    public static void main(String[] args) throws IOException {
        NeuralNetwork SRResNet = new NeuralNetwork()
                .addInputLayer(32, 32, 3)
                .addLayer(new ConvolutionLayer(32, 5, 1, 2))
                .addLayer(new ParametricReLULayer3D())
                .addLayer(new ResidualBlock()
                        .addResidualUnit(new ResidualUnit())
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(getResBlock())
                                .addLayer(getResBlock())
                                .addLayer(getResBlock())
                                .addLayer(getResBlock())
                                .addLayer(getResBlock())
                                .addLayer(getResBlock())
                                .addLayer(getResBlock())
                                .addLayer(getResBlock())
                                .addLayer(new ConvolutionLayer(32, 3, 1, 1))
                                .addLayer(new BatchNormalizationLayer3D(0.8))
                        )
                )
                .addLayer(new ConvolutionLayer(32, 3, 1, 1))
                .addLayer(new PixelShufflerLayer(2))
                .addLayer(new ParametricReLULayer3D())
                .addLayer(new ConvolutionLayer(3, 5, 1, 2))
                .addLayer(new ActivationLayer3D(new FunctionActivation.Sigmoid()))
                .setFunctionLoss(new FunctionLoss.MSE())
                .setOptimizer(new AdamOptimizer(0.0001))
                .create();

        NeuralNetwork discriminator = new NeuralNetwork()
                .addInputLayer(64, 64, 3)
                .addLayer(new ConvolutionLayer(32, 3, 1, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.LeakyReLU(0.2)))
                .addLayer(new ConvolutionLayer(32, 3, 2, 1))
                .addLayer(new BatchNormalizationLayer3D(0.8))
                .addLayer(new ActivationLayer3D(new FunctionActivation.LeakyReLU(0.2)))
                .addLayer(new ConvolutionLayer(64, 3, 1, 1))
                .addLayer(new BatchNormalizationLayer3D(0.8))
                .addLayer(new ActivationLayer3D(new FunctionActivation.LeakyReLU(0.2)))
                .addLayer(new ConvolutionLayer(64, 3, 2, 1))
                .addLayer(new BatchNormalizationLayer3D(0.8))
                .addLayer(new ActivationLayer3D(new FunctionActivation.LeakyReLU(0.2)))
                .addLayer(new ConvolutionLayer(128, 3, 1, 1))
                .addLayer(new BatchNormalizationLayer3D(0.8))
                .addLayer(new ActivationLayer3D(new FunctionActivation.LeakyReLU(0.2)))
                .addLayer(new ConvolutionLayer(128, 3, 2, 1))
                .addLayer(new BatchNormalizationLayer3D(0.8))
                .addLayer(new ActivationLayer3D(new FunctionActivation.LeakyReLU(0.2)))
                .addLayer(new ConvolutionLayer(256, 3, 1, 1))
                .addLayer(new BatchNormalizationLayer3D(0.8))
                .addLayer(new ActivationLayer3D(new FunctionActivation.LeakyReLU(0.2)))
                .addLayer(new ConvolutionLayer(256, 3, 2, 1))
                .addLayer(new BatchNormalizationLayer3D(0.8))
                .addLayer(new ActivationLayer3D(new FunctionActivation.LeakyReLU(0.2)))
                .addLayer(new Flatten3DLayer())
                .addDenseLayer(256, new FunctionActivation.LeakyReLU(0.2))
                .addDenseLayer(1, new FunctionActivation.Sigmoid())
                .setFunctionLoss(new FunctionLoss.BinaryCrossEntropy())
                .setOptimizer(new AdamOptimizer(0.0001))
                .create();

        SRResNet.info();
        discriminator.info();

        ScaleImageNetLoader3D loader = new ScaleImageNetLoader3D(32, 64);

        SRGAN srgan = new SRGAN(SRResNet, discriminator);

        int n = 10;
        for (int i = 0; i < 10000; i++) {
            long start = System.nanoTime();
            if (i % n == 0) {
                ImgNNData3D data = loader.getNextTestData(1);
                ImageCreator.drawColorImage((NNTensor) data.getInput()[0], 32, 32, i / n + "_input", "D:/NetworkTest/SRGAN");
                ImageCreator.drawColorImage((NNTensor) SRResNet.query(data.getInput())[0], 64, 64, i / n + "_test", "D:/NetworkTest/SRGAN");
                ImageCreator.drawColorImage((NNTensor) data.getOutput()[0], 64, 64, i / n + "_real", "D:/NetworkTest/SRGAN");

                SRResNet.save(new FileWriter(new File("D:/NetworkTest/SRGAN/res_generator.txt")));
                discriminator.save(new FileWriter(new File("D:/NetworkTest/SRGAN/discriminator.txt")));
            }
            ImgNNData3D train = loader.getNextTrainData(32);
            System.out.println(i + " - " + srgan.train(train.getInput(),
                    train.getOutput(),
                    0.001f));
            System.out.println((System.nanoTime() - start) / 1000000);
        }
    }

    public static ResidualBlock getResBlock() {
        ResidualBlock resBlock = new ResidualBlock()
                .addResidualUnit(new ResidualUnit())
                .addResidualUnit(new ResidualUnit()
                        .addLayer(new ConvolutionLayer(32, 3, 1, 1))
                        .addLayer(new BatchNormalizationLayer3D(0.8))
                        .addLayer(new ActivationLayer3D(new FunctionActivation.LeakyReLU()))
                        .addLayer(new ConvolutionLayer(32, 3, 1, 1))
                        .addLayer(new BatchNormalizationLayer3D(0.8))
                );

        return resBlock;
    }
}
