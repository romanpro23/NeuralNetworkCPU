package neural_network.layers.convolution_3d.inception;

import lombok.Getter;
import neural_network.layers.LayersBlock;
import neural_network.layers.NeuralLayer;
import neural_network.layers.convolution_3d.ConvolutionNeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class InceptionBlock extends LayersBlock {
    private int outDepth, outWidth, outHeight;
    private NNArray[] input, error;

    public InceptionBlock() {
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
            super.generateOutput(input);
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        if (layers.isEmpty()) {
            this.input = input;
        } else {
            super.generateTrainOutput(input);
        }
    }

    @Override
    public void generateError(NNArray[] error) {
        if (layers.isEmpty()) {
            this.error = error;
        } else {
            super.generateError(error);
        }
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Inception block\n");
        for (NeuralLayer layer : layers) {
            layer.write(writer);
        }
        writer.write("End\n");
        writer.flush();
    }

    @Override
    public NNArray[] getError() {
        if (layers.isEmpty()) {
            return error;
        } else {
            return super.getError();
        }
    }

    @Override
    public NNArray[] getOutput() {
        if (layers.isEmpty()) {
            return input;
        } else {
            return super.getOutput();
        }
    }

    public InceptionBlock addLayer(NeuralLayer layer) {
        layers.add(layer);

        return this;
    }

    public InceptionBlock setTrainable(boolean trainable) {
        super.setTrainable(trainable);

        return this;
    }

    @Override
    public int info() {
        int countParam = 0;
        System.out.println("            |         Inception block       |             ");
        System.out.println("____________|_______________________________|_____________");
        for (NeuralLayer neuralLayer : layers) {
            countParam += neuralLayer.info();
            System.out.println("____________|_______________|_______________|_____________");
        }
        System.out.println("____________|_______________|_______________|_____________");
        return countParam;
    }

    public static InceptionBlock read(Scanner scanner) {
        InceptionBlock inceptionBlock = new InceptionBlock();
        NeuralLayer.read(scanner, inceptionBlock.layers);

        return inceptionBlock;
    }
}