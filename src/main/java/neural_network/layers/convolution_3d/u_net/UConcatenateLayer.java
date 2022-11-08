package neural_network.layers.convolution_3d.u_net;

import neural_network.layers.NeuralLayer;
import neural_network.layers.convolution_3d.ConvolutionNeuralLayer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class UConcatenateLayer extends ConvolutionNeuralLayer {
    private NeuralLayer inputLayer;
    private int indexLayer;

    @Override
    public void initialize(int[] size) {
        if (size.length != 3) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }

        height = size[0];
        width = size[1];
        depth = size[2];
        outWidth = width;
        outHeight = height;
        outDepth = depth;

        outDepth += inputLayer.size()[2];
    }

    public UConcatenateLayer addLayer(NeuralLayer layer, int indexLayer) {
        layer.addNextLayer(this);
        this.inputLayer = layer;
        this.indexLayer = indexLayer;

        return this;
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isTensor(input);

        this.output = NNArrays.concatTensor(this.input, inputLayer.getOutput());
    }

    @Override
    public void generateError(NNArray[] error) {
        errorNL = NNArrays.isTensor(error);
        this.error = NNArrays.subTensor(errorNL, input, 0);
    }

    @Override
    public int info() {
        System.out.println("U-Net layer\t|  " + height + ",\t" + width + ",\t" + depth + "\t| "
                + height + ",\t" + width + ",\t" + outDepth + "\t|");
        return 0;
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("U concatenate layer\n");
        writer.write(indexLayer + "\n");
        writer.flush();
    }

    @Override
    public NNArray[] getError() {
        return error;
    }

    @Override
    public NNArray[] getErrorNL() {
        return NNArrays.subTensor(errorNL, inputLayer.getOutput(), input[0].size());
    }

    public static UConcatenateLayer read(ArrayList<NeuralLayer> layers, Scanner scanner) {
        int index = Integer.parseInt(scanner.nextLine());
        return new UConcatenateLayer().addLayer(layers.get(index), index);
    }
}
