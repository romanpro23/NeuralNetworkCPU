package neural_network.layers.reshape;

import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class Flatten2DLayer extends NeuralLayer {
    protected int depth, width;
    protected int countNeuron;
    protected NNMatrix[] input;
    protected NNVector[] output;
    protected NNMatrix[] error;
    protected NNVector[] errorNL;

    @Override
    public int[] size() {
        return new int[]{countNeuron};
    }

    @Override
    public void initialize(Optimizer optimizer) {
        //no have initialize elements
    }

    @Override
    public int info() {
        System.out.println("Flatten\t\t|  " + width + ",\t" + depth + "\t\t|  " + countNeuron + "\t\t\t|");
        return 0;
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Flatten layer 2D\n");
        writer.flush();
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 2) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        depth = size[1];
        width = size[0];
        countNeuron = depth * width;
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        input = NNArrays.isMatrix(inputs);
        output = new NNVector[inputs.length];

        for (int i = 0; i < output.length; i++) {
            output[i] = new NNVector(input[i].getData());
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        generateOutput(input);
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        error = new NNMatrix[errors.length];

        for (int i = 0; i < errors.length; i++) {
            error[i] = new NNMatrix(width, depth, errorNL[i].getData());
        }
    }

    public NNVector[] getErrorNextLayer(NNArray[] error) {
        NNVector[] errorNL = NNArrays.isVector(error);

        if (!nextLayers.isEmpty()) {
            for (int i = 0; i < errorNL.length; i++) {
                for (NeuralLayer nextLayer : nextLayers) {
                    errorNL[i].add(nextLayer.getErrorNL()[i]);
                }
            }
        }
        return errorNL;
    }

    @Override
    public NNArray[] getOutput() {
        return output;
    }

    @Override
    public NNArray[] getError() {
        return error;
    }

    public static Flatten2DLayer read(Scanner scanner){
        return new Flatten2DLayer();
    }
}
