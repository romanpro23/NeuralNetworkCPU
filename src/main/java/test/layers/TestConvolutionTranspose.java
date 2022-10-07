package test.layers;

import neural_network.layers.convolution_3d.ConvolutionTransposeLayer;
import nnarrays.NNTensor;
import nnarrays.NNTensor4D;

public class TestConvolutionTranspose {
    public static void main(String[] args) {
        NNTensor[] input = new NNTensor[1];
        input[0] = new NNTensor(3,3,1);
        input[0].fill(1);

        NNTensor[] err = new NNTensor[1];
        err[0] = new NNTensor(6,6,1);
        err[0].fill(1);

        NNTensor4D weight = new NNTensor4D(1, 3, 3, 1);
        weight.fill(1);
        weight.set(2, 2);
        weight.set(4, 2);

        ConvolutionTransposeLayer layer = new ConvolutionTransposeLayer(1, 3, 2, 1);
        layer.initialize(new int[]{3,3,1});

        layer.setWeight(weight);

        layer.generateOutput(input);
        layer.generateError(err);
        NNTensor[] out = (NNTensor[]) layer.getOutput();

        for (int i = 0; i < out[0].getRows(); i++) {
            for (int j = 0; j < out[0].getColumns(); j++) {
                System.out.print(out[0].get(i, j, 0) + " ");
            }
            System.out.println();
        }

        System.out.println();
        System.out.println("Error");
        NNTensor errr = (NNTensor) layer.getError()[0];

        for (int i = 0; i < errr.getRows(); i++) {
            for (int j = 0; j < errr.getColumns(); j++) {
                System.out.print(errr.get(i, j, 0) + " ");
            }
            System.out.println();
        }

        System.out.println();
        NNTensor4D errrr = layer.getDerWeight();

        for (int i = 0; i < errrr.getLength(); i++) {
            for (int j = 0; j < errrr.getRow(); j++) {
                System.out.print(errrr.get(0, i, j, 0) + " ");
            }
            System.out.println();
        }
    }
}
