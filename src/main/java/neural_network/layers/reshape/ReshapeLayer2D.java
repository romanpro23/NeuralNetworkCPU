package neural_network.layers.reshape;

import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.*;
import utilities.Use;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class ReshapeLayer2D extends NeuralLayer {
    private final int depth;
    private final int width;
    protected int countNeuron;

    protected NNMatrix[] output;
    protected NNVector[] input;
    protected NNMatrix[] errorNL;
    protected NNVector[] error;

    public ReshapeLayer2D(int width, int depth) {
        this.depth = depth;
        this.width = width;
    }

    @Override
    public int[] size() {
        return new int[]{width, depth};
    }

    @Override
    public void initialize(Optimizer optimizer) {
        //no have initialize elements
    }

    @Override
    public int info() {
        System.out.println("Reshape\t\t|  " + countNeuron + "\t\t\t|  " + width + ",\t" + depth + "\t\t|");
        return 0;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Reshape layer 2D\n");
        writer.write(width + " " + depth + "\n");
        writer.flush();
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 1) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        countNeuron = size[0];
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        input = NNArrays.isVector(inputs);
        output = new NNMatrix[inputs.length];

        if (Use.CPU) {
            for (int i = 0; i < output.length; i++) {
                output[i] = new NNMatrix(width, depth, input[i].getData(), input[i].getSdata());
            }
        }

        if (Use.GPU) {
            for (int i = 0; i < output.length; i++) {
                output[i] = new NNMatrix(width, depth);
                output[i].copy(input[i]);
            }
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        generateOutput(input);
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        error = new NNVector[errors.length];

        if (Use.CPU) {
            for (int i = 0; i < errors.length; i++) {
                error[i] = new NNVector(errorNL[i].getData(), errorNL[i].getSdata());
            }
        }

        if (Use.GPU) {
            for (int i = 0; i < errors.length; i++) {
                error[i] = new NNVector(errorNL[i].size());
                error[i].copy(errorNL[i]);
            }
        }
    }

    public NNMatrix[] getErrorNextLayer(NNArray[] error) {
        NNMatrix[] errorNL = NNArrays.isMatrix(error);

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

    public static ReshapeLayer2D read(Scanner scanner) {
        int[] param = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
        return new ReshapeLayer2D(param[0], param[1]);
    }
}
