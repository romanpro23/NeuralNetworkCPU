package test.utilitas;

import data.ImageCreator;
import data.ciraf.Ciraf10Loader3D;
import data.network_train.NNData;
import lombok.SneakyThrows;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import java.util.Scanner;

public class TestVGG {
    public static void main(String[] args) throws Exception {
        NeuralNetwork vgg = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/Imagenet/vgg8.txt")))
                .setFunctionLoss(new FunctionLoss.MSE())
                .setOptimizer(new AdamOptimizer())
                .create();

        vgg.info();

        NNTensor cat = loadImage("D:/cat.jpg");

//        Ciraf10Loader3D loader = new Ciraf10Loader3D();
//
//        for (int i = 0; i < 100; i++) {
//            NNData data = loader.getNextTestData(1);
//            NNVector result = (NNVector) vgg.query(data.getInput())[0];
//            NNVector real = (NNVector) data.getOutput()[0];
//            System.out.println(i + " " + result.indexMaxElement() + " " + real.indexMaxElement());
//            ImageCreator.drawColorImage((NNTensor) data.getInput()[0], 32, 32, "test_" + i, "D:/NetworkTest/ST");
//        }

        NNVector result = (NNVector) vgg.query(new NNTensor[]{cat})[0];
        System.out.println("Result " + Arrays.toString(result.indexMaxElement(5)));
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
