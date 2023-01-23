package neural_network.layers.layer_3d.residual;

import neural_network.layers.LayersBlock;
import neural_network.layers.NeuralLayer;
import nnarrays.NNArray;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class ResidualUnit extends LayersBlock {
    private NNArray[] input, error;

    public ResidualUnit() {
        super();
    }

    @Override
    public int[] size() {
        if(this.layers.isEmpty()) {
            return inputSize;
        }
        return super.size();
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 3) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        inputSize = size;
        if (!layers.isEmpty()) {
            super.initialize(size);
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
        writer.write("Residual unit\n");
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

    public ResidualUnit addLayer(NeuralLayer layer) {
        layers.add(layer);

        return this;
    }

    public ResidualUnit setTrainable(boolean trainable) {
        super.setTrainable(trainable);

        return this;
    }

    @Override
    public int info() {
        int countParam = 0;
        System.out.println("            |          Residual unit        |             ");
        System.out.println("____________|_______________________________|_____________");
        for (NeuralLayer neuralLayer : layers) {
            countParam += neuralLayer.info();
            System.out.println("____________|_______________|_______________|_____________");
        }
        return countParam;
    }

    public static ResidualUnit read(Scanner scanner) {
        ResidualUnit inceptionBlock = new ResidualUnit();
        NeuralLayer.read(scanner, inceptionBlock.layers);

        return inceptionBlock;
    }
}