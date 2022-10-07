package neural_network.layers.convolution_3d.densely;

import neural_network.layers.LayersBlock;
import neural_network.layers.NeuralLayer;
import nnarrays.NNArray;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class DenseBlock extends LayersBlock {
    private int outDepth, outWidth, outHeight;
    private NNArray[] input;

    public DenseBlock() {
        super();
    }

    @Override
    public int[] size() {
        return new int[]{outHeight, outWidth, outDepth};
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 3) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        inputSize = size;
        if (!layers.isEmpty()) {
            super.initialize(size);

            outDepth = layers.get(layers.size() - 1).size()[2];
            outHeight = layers.get(layers.size() - 1).size()[1];
            outWidth = layers.get(layers.size() - 1).size()[0];
        } else {
            outDepth = inputSize[2];
            outHeight = inputSize[0];
            outWidth = inputSize[1];
        }
    }

    @Override
    public void generateOutput(NNArray[] input) {
        if (layers.isEmpty()) {
            this.input = input;
        } else {
            this.input = input;
            super.generateOutput(input);
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        if (layers.isEmpty()) {
            this.input = input;
        } else {
            this.input = input;
            super.generateTrainOutput(input);
        }
    }

    @Override
    public void generateError(NNArray[] error) {
        super.generateError(error);
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Dense block\n");
        for (NeuralLayer layer : layers) {
            layer.write(writer);
        }
        writer.write("End\n");
        writer.flush();
    }

    @Override
    public NNArray[] getError() {
        return super.getError();
    }

    @Override
    public NNArray[] getOutput() {
        return super.getOutput();
    }

    public NNArray[] getInput() {
        return input;
    }

    public DenseBlock addLayer(NeuralLayer layer) {
        layers.add(layer);

        return this;
    }

    public DenseBlock setTrainable(boolean trainable) {
        super.setTrainable(trainable);

        return this;
    }

    @Override
    public int info() {
        int countParam = 0;
        System.out.println("            |           Dense block         |             ");
        System.out.println("____________|_______________________________|_____________");
        for (NeuralLayer neuralLayer : layers) {
            countParam += neuralLayer.info();
            System.out.println("____________|_______________|_______________|_____________");
        }
        System.out.println("____________|_______________|_______________|_____________");
        return countParam;
    }

    public static DenseBlock read(Scanner scanner) {
        DenseBlock inceptionBlock = new DenseBlock();
        NeuralLayer.read(scanner, inceptionBlock.layers);

        return inceptionBlock;
    }
}