package test.layers;

import neural_network.layers.convolution_3d.ConvolutionLayer;
import neural_network.layers.convolution_3d.ConvolutionTransposeLayer;
import neural_network.layers.convolution_3d.GroupedConvolutionLayer;
import nnarrays.NNTensor;
import nnarrays.NNTensor4D;

public class TestConvolutionTranspose {
    public static void main(String[] args) {
        NNTensor[] input = new NNTensor[1];
        input[0] = new NNTensor(3,3,2);
        input[0].fill(1);
        input[0].set(1, 1, 1, 3);
        input[0].set(1, 2, 2, 2);

        NNTensor[] err = new NNTensor[1];
        err[0] = new NNTensor(3,3,4);
        err[0].fill(1);
        err[0].set(2, 2, 2, 2);
        err[0].set(2, 1, 0, 3);

        NNTensor4D weight = new NNTensor4D(4, 3, 3, 2);
        weight.fill(1);
        weight.set(2, 2);
        weight.set(4, 2);

        GroupedConvolutionLayer layer = new GroupedConvolutionLayer(4, 3, 1, 1, 1);
        layer.initialize(new int[]{3,3,2});

        layer.setWeight(weight);

        layer.generateOutput(input);
        layer.generateError(err);
        NNTensor[] out = (NNTensor[]) layer.getOutput();
        System.out.println("Output");
        for (int d = 0; d < out[0].getDepth(); d++) {
            for (int i = 0; i < out[0].getRows(); i++) {
                for (int j = 0; j < out[0].getColumns(); j++) {
                    System.out.print(out[0].get(i, j, d) + " ");
                }
                System.out.println();
            }
            System.out.println();
        }


        System.out.println();
        System.out.println("Error");
        NNTensor errr = (NNTensor) layer.getError()[0];

        for (int d = 0; d < errr.getDepth(); d++) {
            for (int i = 0; i < errr.getRows(); i++) {
                for (int j = 0; j < errr.getColumns(); j++) {
                    System.out.print(errr.get(i, j, d) + " ");
                }
                System.out.println();
            }
            System.out.println();
        }

        System.out.println();
        System.out.println("Weight");
        NNTensor4D errrr = layer.getDerWeight();

        for (int d = 0; d < errrr.depth(); d++) {
            for (int c = 0; c < errrr.column(); c++) {
                for (int i = 0; i < errrr.getLength(); i++) {
                    for (int j = 0; j < errrr.getRow(); j++) {
                        System.out.print(errrr.get(0, i, j, 0) + " ");
                    }
                    System.out.println();
                }
                System.out.println();
            }
            System.out.println();
        }

    }
}
