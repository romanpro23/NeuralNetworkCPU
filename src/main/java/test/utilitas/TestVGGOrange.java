package test.utilitas;

import data.ImageCreator;
import data.imageNet.TinyImageNetLoader3D;
import data.loaders.TransformData;
import lombok.SneakyThrows;
import neural_network.initialization.Initializer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNTensor;
import nnarrays.NNVector;
import trainer.DataMetric;
import trainer.DataTrainer;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import java.util.Scanner;

public class TestVGGOrange {
    public static void main(String[] args) throws Exception {
        NeuralNetwork vgg = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/Imagenet/vgg8.txt")))
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .setOptimizer(new AdamOptimizer())
                .create();

        vgg.setTrainable(false);
        vgg.info();

        NNTensor orange = loadImage("D:/cat.jpg");
        NNTensor delta = new NNTensor(64, 64, 3);
        AdamOptimizer optimizer = new AdamOptimizer(0.1);
        optimizer.addDataOptimize(orange, delta);

        NNVector result = new NNVector(200);
        result.set(32, 1f);
        for (int i = 0; i < 100; i++) {
            /*orange.clip(-1, 1);
            ImageCreator.drawColorImage(orange, 64, 64, "test_orange_" + i, "D:/NetworkTest/ST", true);
            System.out.println(vgg.train(new NNTensor[]{orange}, new NNVector[]{result}, false));
            System.out.println(Arrays.toString(vgg.getOutputs()[0].indexMaxElement(5)));
            delta.clear();
            delta.add(vgg.getError()[0]);
            optimizer.update();*/
        }
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
                input.set(i1, j1, 0, color.getRed() / 255f * 2 - 1);
                input.set(i1, j1, 1, color.getGreen() / 255f * 2 - 1);
                input.set(i1, j1, 2, color.getBlue() / 255f * 2 - 1);
            }
        }

        return input;
    }
}
