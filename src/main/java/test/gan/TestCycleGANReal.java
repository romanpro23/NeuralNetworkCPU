package test.gan;

import data.ImageCreator;
import data.image2image.AppleToOrangeLoader3D;
import data.loaders.TransformData;
import lombok.SneakyThrows;
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
import org.imgscalr.Scalr;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.util.Scanner;

public class TestCycleGANReal {
    public static void main(String[] args) throws Exception {
//        NeuralNetwork discriminatorA = getDiscriminator();
//
//        NeuralNetwork discriminatorB = getDiscriminator();
//
//        NeuralNetwork generatorA = getGenerator();
//
//        NeuralNetwork generatorB = getGenerator();
        NeuralNetwork generatorA = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/CycleGAN/res_generator_orange.txt")))
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0002))
                .setFunctionLoss(new FunctionLoss.MAE())
                .create();

        NeuralNetwork generatorB = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/CycleGAN/res_generator_apple.txt")))
                .setOptimizer(new AdamOptimizer(0.5, 0.999, 0.0002))
                .setFunctionLoss(new FunctionLoss.MAE())
                .create();

        System.out.println(generatorA.getLayers().size());

        generatorB.info();

        NNTensor[] apples = new NNTensor[]{loadImage("D:/apple.jpg")};
        NNTensor[] oranges = new NNTensor[]{loadImage("D:/orange.jpg")};

        ImageCreator.drawColorImage((NNTensor) generatorA.query(apples)[0], 64, 64, "apple_orange", "D:", true);
        ImageCreator.drawColorImage((NNTensor) generatorB.query(generatorA.getOutputs())[0], 64, 64, "apple_recon", "D:", true);
        ImageCreator.drawColorImage((NNTensor) generatorA.query(generatorB.getOutputs())[0], 64, 64, "apple_orange_recon", "D:", true);

        ImageCreator.drawColorImage((NNTensor) generatorB.query(oranges)[0], 64, 64, "orange_apple", "D:", true);
        ImageCreator.drawColorImage((NNTensor) generatorA.query(generatorB.getOutputs())[0], 64, 64, "orange_recon", "D:", true);
        ImageCreator.drawColorImage((NNTensor) generatorB.query(generatorA.getOutputs())[0], 64, 64, "orange_apple_recon", "D:", true);
    }

    static int sizeImage = 64;

    @SneakyThrows
    private static NNTensor loadImage(String str) {
        NNTensor input = new NNTensor(sizeImage, sizeImage, 3);
        BufferedImage image = ImageIO.read(new File(str));

        for (int i = 0; i < sizeImage; i++) {
            for (int j = 0; j < sizeImage; j++) {
                Color color = new Color(image.getRGB(i, j));
                int i1 = i;
                int j1 = j;
                input.set(i1, j1, 0, color.getRed() * 2 - 1);
                input.set(i1, j1, 1, color.getGreen() * 2 - 1);
                input.set(i1, j1, 2, color.getBlue() * 2 - 1);
            }
        }

        return input;
    }
}
