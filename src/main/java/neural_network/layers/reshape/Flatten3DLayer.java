package neural_network.layers.reshape;

import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class Flatten3DLayer extends NeuralLayer {
    protected int depth, height, width;
    protected int countNeuron;
    protected NNTensor[] input;
    protected NNVector[] output;
    protected NNTensor[] error;
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
        System.out.println("Flatten\t\t|  " + height + ",\t" + width + ",\t" + depth + "\t|  " + countNeuron + "\t\t\t|");
        return 0;
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Flatten layer 3D\n");
        writer.flush();
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 3) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        depth = size[2];
        height = size[0];
        width = size[1];
        countNeuron = depth * height * width;
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        input = NNArrays.isTensor(inputs);
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
        error = new NNTensor[errors.length];

        for (int i = 0; i < errors.length; i++) {
            error[i] = new NNTensor(height, width, depth, errorNL[i].getData());
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

    public static Flatten3DLayer read(Scanner scanner){
        return new Flatten3DLayer();
    }
}
