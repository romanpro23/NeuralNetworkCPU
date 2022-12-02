package test.utilitas;

import data.ImageCreator;
import lombok.SneakyThrows;
import neural_network.activation.FunctionActivation;
import neural_network.layers.convolution_3d.ActivationLayer3D;
import neural_network.layers.convolution_3d.ConvolutionLayer;
import neural_network.layers.convolution_3d.UpSamplingLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.style_transfer.AdaINStyleTransfer;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNTensor;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Scanner;

public class StyleTransferTestAdaIN {
    public static void main(String[] args) throws Exception {
        NeuralNetwork vgg = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/ciraf/vgg10.txt")));

        vgg.addInputLayer(224, 224, 3);
        for (int i = vgg.getLayers().size() - 1; i > 18; i--) {
            vgg.getLayers().remove(i);
        }
        vgg.setTrainable(false).create();
        vgg.info();

        NeuralNetwork decoder = new NeuralNetwork()
                .addInputLayer(28, 28, 256)
                .addLayer(new ConvolutionLayer(128, 3, 1, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new UpSamplingLayer(2))
                .addLayer(new ConvolutionLayer(128, 3, 1, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new ConvolutionLayer(128, 3, 1, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new ConvolutionLayer(64, 3, 1, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new UpSamplingLayer(2))
                .addLayer(new ConvolutionLayer(64, 3, 1, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new ConvolutionLayer(32, 3, 1, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new UpSamplingLayer(2))
                .addLayer(new ConvolutionLayer(32, 3, 1, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new ConvolutionLayer(3, 3, 1, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.Sigmoid()))
                .setFunctionLoss(new FunctionLoss.MSE())
                .setOptimizer(new AdamOptimizer())
                .create();

        decoder.info();

        NNTensor content = loadImage("D:/content.jpg");
        NNTensor style = loadImage("D:/style.jpg");

        ImageCreator.drawColorImage(content, 224, 224, "_content", "D:/NetworkTest/ST");
        ImageCreator.drawColorImage(style, 224, 224, "_style", "D:/NetworkTest/ST");

        AdaINStyleTransfer styleTransfer = new AdaINStyleTransfer(vgg, decoder)
                .setContent(content)
                .setStyle(style)
                .create();

        for (int i = 0; i < 10000; i++) {
            System.out.println(styleTransfer.train());
            ImageCreator.drawColorImage(styleTransfer.getResult()[0], 224, 224, "result_" + i, "D:/NetworkTest/ST");
        }
    }

    @SneakyThrows
    private static NNTensor loadImage(String str) {
        int sizeImage = 224;
        NNTensor input = new NNTensor(sizeImage, sizeImage, 3);
        BufferedImage image = ImageIO.read(new File(str));

        for (int i = 0; i < sizeImage; i++) {
            for (int j = 0; j < sizeImage; j++) {
                Color color = new Color(image.getRGB(i, j));
                int i1 = i;
                int j1 = j;
                input.set(i1, j1, 0, color.getRed() / 255f);
                input.set(i1, j1, 1, color.getGreen() / 255f);
                input.set(i1, j1, 2, color.getBlue() / 255f);
            }
        }

        return input;
    }
}
