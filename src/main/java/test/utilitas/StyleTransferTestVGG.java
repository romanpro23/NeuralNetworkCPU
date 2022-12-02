package test.utilitas;

import data.ImageCreator;
import lombok.SneakyThrows;
import neural_network.layers.convolution_3d.BatchNormalizationLayer3D;
import neural_network.layers.convolution_3d.DropoutLayer3D;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.style_transfer.StyleTransfer;
import neural_network.optimizers.AdamOptimizer;
import neural_network.optimizers.Optimizer;
import neural_network.optimizers.SGDOptimizer;
import nnarrays.NNMatrix;
import nnarrays.NNTensor;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Scanner;

public class StyleTransferTestVGG {
    public static void main(String[] args) throws Exception {
        NeuralNetwork vgg = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/Imagenet/vgg16.txt")))
//                .read(new Scanner(new File("D:/NetworkTest/ciraf/vgg13.txt")))
                .addInputLayer(224, 224, 3)
                .removeLastLayers(10)
                .setTrainable(false)
                .create();

        vgg.info();
        NNTensor content = loadImage("D:/content.jpg");
        NNTensor style = loadImage("D:/style.jpg");

        ImageCreator.drawColorImage(content, 224, 224, "_content", "D:/NetworkTest/ST", true);
//        ImageCreator.drawImage(content, 224, 224, "r_content", "D:/NetworkTest/ST", 0);
//        ImageCreator.drawImage(content, 224, 224, "g_content", "D:/NetworkTest/ST", 1);
//        ImageCreator.drawImage(content, 224, 224, "b_content", "D:/NetworkTest/ST", 2);
        ImageCreator.drawColorImage(style, 224, 224, "_style", "D:/NetworkTest/ST", true);

        StyleTransfer styleTransfer = new StyleTransfer(vgg)
                .addStyleLayer(
//                        vgg.getLayers().get(1),
//                        vgg.getLayers().get(6),
//                        vgg.getLayers().get(11),
                        vgg.getLayers().get(17)
                )
                .setContent(content)
                .setStyle(style)
                .setOptimizer(new AdamOptimizer(0.1))
                .create();

        vgg.query(new NNTensor[]{content});
//        NNTensor contentVGG = (NNTensor) vgg.getLayers().get(1).getOutput()[0];
        NNTensor contentVGG = (NNTensor) vgg.getOutputs()[0];
        contentVGG.clip(0, 1);

        int size = contentVGG.getRows(), depth = Math.min(contentVGG.getDepth(), 64);
        for (int i = 0; i < depth; i++) {
            ImageCreator.drawImage(contentVGG, size, size, i + "_content_vgg", "D:/NetworkTest/ST", i);
        }

        for (int i = 0; i < 10000; i++) {
            ImageCreator.drawColorImage(styleTransfer.getResult(), 224, 224, "result_" + i, "D:/NetworkTest/ST", true);
            System.out.println(styleTransfer.train(1, 1));
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
                input.set(i1, j1, 0, color.getRed() / 255f * 2 - 1f);
                input.set(i1, j1, 1, color.getGreen() / 255f * 2 - 1);
                input.set(i1, j1, 2, color.getBlue() / 255f * 2 - 1);
            }
        }

        return input;
    }
}
