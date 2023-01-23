package test.style_transfer;

import data.ImageCreator;
import neural_network.layers.layer_3d.ActivationLayer3D;
import neural_network.layers.layer_3d.MaxPoolingLayer;
import neural_network.network.NeuralNetwork;
import neural_network.network.style_transfer.StyleTransfer;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNTensor;

import java.io.File;
import java.util.Arrays;
import java.util.Scanner;

public class StyleTransferVGG {

    static int sizeImage = 64;
    public static void main(String[] args) throws Exception {
        NeuralNetwork vgg = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/Imagenet/vgg16.txt")))
                .addInputLayer(sizeImage, sizeImage, 3)
//                .removeLastLayers(9)
                .setTrainable(false)
                .create();

        vgg.info();
        NNTensor content = ImageCreator.loadImage("D:/NetworkTest/Photo/cat.jpg", sizeImage);
        NNTensor style = ImageCreator.loadImage("D:/NetworkTest/Photo/style.jpg", sizeImage);

        ImageCreator.drawColorImage(content, sizeImage, sizeImage, "_content", "D:/NetworkTest/ST", true);
        ImageCreator.drawColorImage(style, sizeImage, sizeImage, "_style", "D:/NetworkTest/ST", true);
        ImageCreator.drawColorImage(ImageCreator.horizontalSobelEdge(content), sizeImage, sizeImage, "_horizontal", "D:/NetworkTest/ST", true);
        ImageCreator.drawColorImage(ImageCreator.verticalSobelEdge(content), sizeImage, sizeImage, "_vertical", "D:/NetworkTest/ST", true);
        ImageCreator.drawColorImage(ImageCreator.diagonalSobelEdge(content), sizeImage, sizeImage, "_diagonal", "D:/NetworkTest/ST", true);

        StyleTransfer styleTransfer = new StyleTransfer(vgg)
                .addStyleLayer(
                        vgg.getLayers().get(1),
                        vgg.getLayers().get(6),
                        vgg.getLayers().get(11),
                        vgg.getLayers().get(18)
//                        vgg.getLayers().get(25)
                )
                .addContentLayer(
//                        vgg.getLayer(3)
                        vgg.getLayer(20)
                )
                .setStyle(style)
                .setContent(content)
//                .setResult(ImageCreator.loadImage("D:/NetworkTest/Photo/style2.jpg", 224))
                .setOptimizer(new AdamOptimizer(0.99, 0.999, 0.1))
//                .setOptimizer(new AdaBeliefOptimizer(0.99, 0.999, 0.1))
//                .setOptimizer(new MomentumOptimizer(100,0.9))
                .create();

        NNTensor contentVGG;
        int size, depth;

        for (int j = 0; j < vgg.size(); j++) {
            if(vgg.getLayer(j) instanceof ActivationLayer3D || vgg.getLayer(j) instanceof MaxPoolingLayer) {
                contentVGG = (NNTensor) vgg.getLayer(j).getOutput()[0];
                size = contentVGG.getRows();
                depth = Math.min(contentVGG.getDepth(), 64);
                for (int i = 0; i < depth; i++) {
                    ImageCreator.drawImage(contentVGG, size, size, j + "_" + i + "_vgg_", "D:/NetworkTest/ST", i);
                }
            }
        }

        System.out.println(Arrays.toString(vgg.getLayer(28).getOutput()[0].getData()));
        System.out.println(Arrays.toString(vgg.getLayer(29).getOutput()[0].getData()));
        System.out.println(Arrays.toString(vgg.getLayer(30).getOutput()[0].getData()));

        for (int i = 0; i < 100; i++) {
            styleTransfer.getResult().clip(-1, 1);
            ImageCreator.drawColorImage(styleTransfer.getResult(), sizeImage, sizeImage, "result_" + i, "D:/NetworkTest/ST", true);
            System.out.println(styleTransfer.train(1, 1000));
        }
    }
}
