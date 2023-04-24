package test.style_transfer;

import data.ImageCreator;
import neural_network.layers.layer_3d.ActivationLayer3D;
import neural_network.layers.layer_3d.ConvolutionLayer;
import neural_network.layers.layer_3d.MaxPoolingLayer;
import neural_network.network.NeuralNetwork;
import neural_network.network.style_transfer.StyleTransfer;
import neural_network.optimizers.*;
import nnarrays.NNTensor;
import nnarrays.NNTensor4D;

import java.io.File;
import java.util.Arrays;
import java.util.Scanner;

public class StyleTransferAlexNet {

    static int sizeImage = 64;
    public static void main(String[] args) throws Exception {
        NeuralNetwork alexnet = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/Imagenet/alexnet_norm.txt")))
                .addInputLayer(sizeImage, sizeImage, 3)
//                .removeLastLayers(12)
                .setTrainable(false)
                .create();

        alexnet.info();
        NNTensor content = ImageCreator.loadImage("D:/NetworkTest/Photo/labrador.jpg", sizeImage);
        NNTensor style = ImageCreator.loadImage("D:/NetworkTest/Photo/style.jpg", sizeImage);

        ImageCreator.drawColorImage(content, sizeImage, sizeImage, "_content", "D:/NetworkTest/ST", true);
        ImageCreator.drawColorImage(style, sizeImage, sizeImage, "_style", "D:/NetworkTest/ST", true);
        ImageCreator.drawColorImage(ImageCreator.horizontalSobelEdge(content), sizeImage, sizeImage, "_horizontal", "D:/NetworkTest/ST", true);
        ImageCreator.drawColorImage(ImageCreator.verticalSobelEdge(content), sizeImage, sizeImage, "_vertical", "D:/NetworkTest/ST", true);
        ImageCreator.drawColorImage(ImageCreator.diagonalSobelEdge(content), sizeImage, sizeImage, "_diagonal", "D:/NetworkTest/ST", true);

        StyleTransfer styleTransfer = new StyleTransfer(alexnet)
                .addStyleLayer(
                        alexnet.getLayers().get(1),
                        alexnet.getLayers().get(4),
                        alexnet.getLayers().get(7)
//                        alexnet.getLayers().get(11)
                )
                .addContentLayer(
                        alexnet.getLayer(9),
                        alexnet.getLayer(2)
                )
                .setStyle(style)
                .setContent(content)
                .setOptimizer(new AdamOptimizer(0.99, 0.999, 0.02))
                .create();

        NNTensor contentVGG;
        int size, depth;

        for (int j = 0; j < alexnet.size(); j++) {
            if(alexnet.getLayer(j) instanceof ActivationLayer3D || alexnet.getLayer(j) instanceof MaxPoolingLayer) {
                contentVGG = (NNTensor) alexnet.getLayer(j).getOutput()[0];
                size = contentVGG.getRows();
                depth = Math.min(contentVGG.getDepth(), 64);
                for (int i = 0; i < depth; i++) {
                    ImageCreator.drawImage(contentVGG, size, size, j + "_" + i + "_alexnet", "D:/NetworkTest/ST", i);
                }
            }
        }

//        System.out.println(Arrays.toString(alexnet.getLayer(11).getOutput()[0].getData()));
//        System.out.println(Arrays.toString(alexnet.getLayer(12).getOutput()[0].getData()));
//        System.out.println(Arrays.toString(alexnet.getLayer(13).getOutput()[0].getData()));
//        System.out.println(Arrays.toString(alexnet.getLayer(14).getOutput()[0].getData()));

        for (int i = 0; i < 100; i++) {
            styleTransfer.getResult().clip(-1, 1);
            ImageCreator.drawColorImage(styleTransfer.getResult(), sizeImage, sizeImage, "result_" + i, "D:/NetworkTest/ST", true);
            System.out.println(styleTransfer.train(1, 10000));
        }
    }
}
