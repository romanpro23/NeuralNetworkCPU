package neural_network.layers.layer_2d;

import neural_network.layers.LayersBlock;
import neural_network.layers.NeuralLayer;
import nnarrays.NNArray;
import nnarrays.NNMatrix;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class AdditionBlock extends LayersBlock {
    private NNMatrix[] output;
    private NNMatrix[] error;

    private int width, depth;

    public AdditionBlock() {
        super();
    }

    @Override
    public int[] size() {
        return new int[]{width, depth};
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 2) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        width = size[0];
        depth = size[1];
        super.initialize(size);
    }

    @Override
    public void generateOutput(NNArray[] input) {
        super.generateOutput(input);
        output = addition(input, super.getOutput());
    }

    private NNMatrix[] addition(NNArray[] input, NNArray[] outputs) {
        NNMatrix[] output = new NNMatrix[input.length];

        for (int i = 0; i < input.length; i++) {
            output[i] = new NNMatrix(width, depth, input[i].isTYPE());
            output[i].add(input[i]);
            output[i].add(outputs[i]);
        }

        return output;
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        super.generateTrainOutput(input);
        output = addition(input, super.getOutput());
    }

    @Override
    public void generateError(NNArray[] error) {
        super.generateError(error);
        this.error = addition(error, super.getError());
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Additional block\n");
        for (NeuralLayer layer : layers) {
            layer.save(writer);
        }
        writer.write("End\n");
        writer.flush();
    }

    @Override
    public NNArray[] getError() {
        return error;
    }

    @Override
    public NNArray[] getOutput() {
        return output;
    }

    public AdditionBlock addLayer(NeuralLayer layer) {
        layers.add(layer);

        return this;
    }

    public AdditionBlock setTrainable(boolean trainable) {
        super.setTrainable(trainable);

        return this;
    }

    @Override
    public int info() {
        int countParam = 0;
        System.out.println("            |         Addition block        |             ");
        System.out.println("____________|_______________________________|_____________");
        for (NeuralLayer neuralLayer : layers) {
            countParam += neuralLayer.info();
            System.out.println("____________|_______________|_______________|_____________");
        }
        System.out.println("            | " + width + ",\t" + depth + "\t\t| " + width + ",\t" + depth + "\t\t|\t" + countParam);

        return countParam;
    }

    public static AdditionBlock read(Scanner scanner) {
        AdditionBlock block = new AdditionBlock();
        NeuralLayer.read(scanner, block.layers);

        return block;
    }
}