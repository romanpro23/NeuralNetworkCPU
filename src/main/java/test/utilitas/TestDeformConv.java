package test.utilitas;

import data.ImageCreator;
import data.mnist.MNISTLoader3D;
import neural_network.layers.layer_3d.ActivationLayer3D;
import neural_network.layers.layer_3d.DeformableV2ConvolutionLayer;
import neural_network.layers.layer_3d.MaxPoolingLayer;
import neural_network.network.NeuralNetwork;
import nnarrays.NNTensor;

import java.io.File;
import java.util.Scanner;

public class TestDeformConv {
    static int sizes = 28;

    public static void main(String[] args) throws Exception {
        MNISTLoader3D loader = new MNISTLoader3D();

        NeuralNetwork network = NeuralNetwork.read(new Scanner(new File("modulatedV2_conv.txt"))).create();

        NNTensor input = (NNTensor) loader.getNextTestData(1).getInput()[0];
        System.out.println(network.query(input).indexMaxElement());

        ImageCreator.drawImage(input, sizes, sizes, "_input", "D:/NetworkTest/DeformConv", 0);

        NNTensor contentVGG;
        int size, depth;

        DeformableV2ConvolutionLayer layer = null;
        for (int j = 0; j < network.size(); j++) {
            if (network.getLayer(j) instanceof ActivationLayer3D || network.getLayer(j) instanceof MaxPoolingLayer) {
                contentVGG = (NNTensor) network.getLayer(j).getOutput()[0];
                size = contentVGG.getRows();
                depth = contentVGG.getDepth();
                for (int i = 0; i < depth; i++) {
                    ImageCreator.drawImage(contentVGG, size, size, j + "_" + i + "_number", "D:/NetworkTest/DeformConv", i);
                }
            }
            if (network.getLayer(j) instanceof DeformableV2ConvolutionLayer && layer == null) {
                layer = (DeformableV2ConvolutionLayer) network.getLayer(j);
            }
        }
        NNTensor color = getColorNumber(input);
        ImageCreator.drawColorImage(color, sizes, sizes, "_color", "D:/NetworkTest/DeformConv");

        NNTensor offset = layer.getOffset()[0];
        int center = 14;
        int k = -2, l = -2;
        for (int i = 0; i < offset.getDepth(); i += 2) {
            float y = offset.get(center, center, i);
            float x = offset.get(center, center, i + 1);
            if (l == 3) {
                l = -2;
                k++;
            }
            int y0 = Math.round(y + k + center);
            int x0 = Math.round(x + l+ center);
            if (y0 >= 0 && x0 >= 0 && y0 < 28 && x0 < 28) {
                color.set(y0, x0, 0, 1);
                color.set(y0, x0, 1, 0);
                color.set(y0, x0, 2, 0);
            }
            l++;
        }
        ImageCreator.drawColorImage(color, sizes, sizes, "_color_deform", "D:/NetworkTest/DeformConv");
    }

    static NNTensor getColorNumber(NNTensor tensor) {
        NNTensor result = new NNTensor(tensor.getRows(), tensor.getColumns(), 3);
        for (int i = 0; i < tensor.getRows(); i++) {
            for (int j = 0; j < tensor.getColumns(); j++) {
                result.set(i, j, 0, tensor.get(i, j, 0));
                result.set(i, j, 1, tensor.get(i, j, 0));
                result.set(i, j, 2, tensor.get(i, j, 0));
            }
        }

        return result;
    }
}
