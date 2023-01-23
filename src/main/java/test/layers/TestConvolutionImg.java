package test.layers;

import data.ImageCreator;
import neural_network.initialization.Initializer;
import neural_network.layers.layer_3d.*;
import nnarrays.NNTensor;

import java.io.IOException;

public class TestConvolutionImg {
    public static void main(String[] args) throws IOException {
        String[] strings = new String[]{
                "cat2.jpg",
                "cat.jpg",
                "rabbit.jpg",
                "orange.jpg",
                "orange2.png",
                "tiger.jpg",
                "tiger2.jpg",
                "tiger3.jpg",
                "labrador.jpg"
        };

        NNTensor[] input = new NNTensor[strings.length];
        for (int i = 0; i < strings.length; i++) {
            input[i] = ImageCreator.loadImage("D:/NetworkTest/Photo/" + strings[i], 64);
        }

        ConvolutionLayer layer = new ConvolutionLayer(64, 3, 1, 1).setInitializer(new Initializer.HeNormal());
        layer.initialize(new int[]{64, 64, 3});

//        Optimizer optimizer = new AdamOptimizer();
//        layer.initialize(optimizer);
//        FunctionLoss loss = new FunctionLoss.MSE();
        int size = layer.size()[0];

        layer.generateOutput(input);
        int depth = 64;
        for (int i = 0; i < depth; i++) {
            ImageCreator.drawImage((NNTensor) layer.getOutput()[0], size, size, + i + "_conv", "D:/NetworkTest/DeformConv", i);
        }
//
        for (int i = 0; i < 128; i++) {
            long start = System.nanoTime();
            layer.generateOutput(input);
            System.out.println((System.nanoTime() - start) / 1000000);
            System.out.println((Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024.0));

//            System.out.println(loss.findAccuracy(layer.getOutput(), output));
//            layer.generateError(NNArrays.toTensor(loss.findDerivative(layer.getOutput(), output), 64, 64, 64));
//            optimizer.update();
        }
    }
}
