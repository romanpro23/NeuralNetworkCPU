package test.style_transfer;

import data.ImageCreator;
import lombok.SneakyThrows;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNTensor;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Scanner;

public class StyleTransferTest {
    public static void main(String[] args) throws Exception {
        NeuralNetwork vgg = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/ciraf/vgg10.txt")));


        vgg.addInputLayer(224, 224, 3);
        for (int i = vgg.getLayers().size() - 1; i > 22; i--) {
            vgg.getLayers().remove(i);
        }

        vgg.setFunctionLoss(new FunctionLoss.MSE())
                .setOptimizer(new AdamOptimizer())
                .create();

        vgg.setTrainable(false);
        vgg.info();

        NNTensor content = loadImage("D:/content.jpg");
        NNTensor style = loadImage("D:/style.jpg");

        NNTensor result = new NNTensor(224, 224, 3);
        NNTensor delta = new NNTensor(224, 224, 3);

        result.add(content);
//        new Initializer.RandomNormal(3).initialize(result);
//        result.sigmoid(result);

        ImageCreator.drawColorImage(result, 224, 224, "_result", "D:/NetworkTest/ST");

        AdamOptimizer optimizer = new AdamOptimizer(0.1);

        optimizer.addDataOptimize(result, delta);

        for (int i = 0; i < 10000; i++) {
            float accuracy = 0;
            vgg.query(new NNTensor[]{result});

            vgg.getOutputs()[0].l2norm();
            vgg.getOutputs()[0].mul(-1);
            vgg.train(vgg.getOutputs(), false);

            delta.add(vgg.getError()[0]);
            optimizer.update();

            ImageCreator.drawColorImage(result, 224, 224, "resultDream_" + i, "D:/NetworkTest/ST");
            System.out.println(i + " - " + accuracy);
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
