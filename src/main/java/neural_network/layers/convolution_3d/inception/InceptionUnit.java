package neural_network.layers.convolution_3d.inception;

import neural_network.layers.LayersBlock;
import neural_network.layers.NeuralLayer;
import nnarrays.NNArray;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class InceptionUnit extends LayersBlock {
    private int outDepth, outWidth, outHeight;
    private NNArray[] input, error;

    public InceptionUnit() {
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
    public void save(FileWriter writer) throws IOException {
        writer.write("Inception unit\n");
        for (NeuralLayer layer : layers) {
            layer.save(writer);
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

    public InceptionUnit addLayer(NeuralLayer layer) {
        layers.add(layer);

        return this;
    }

    public InceptionUnit setTrainable(boolean trainable) {
        super.setTrainable(trainable);

        return this;
    }

    @Override
    public int info() {
        int countParam = 0;
        System.out.println("            |         Inception unit        |             ");
        System.out.println("____________|_______________________________|_____________");
        for (NeuralLayer neuralLayer : layers) {
            countParam += neuralLayer.info();
            System.out.println("____________|_______________|_______________|_____________");
        }
        return countParam;
    }

    public static InceptionUnit read(Scanner scanner) {
        InceptionUnit inceptionUnit = new InceptionUnit();
        NeuralLayer.read(scanner, inceptionUnit.layers);

        return inceptionUnit;
    }
}