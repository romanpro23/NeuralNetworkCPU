package neural_network.layers.layer_2d;

import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.*;
import utilities.Use;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

import static utilities.Use.GPU_Sleep;
import static utilities.Use.GPU_WakeUp;

public class Float2Half2D extends NeuralLayer2D {
    protected int depth, width;
    protected int countNeuron;
    protected NNMatrix[] input;
    protected NNMatrix[] output;
    protected NNMatrix[] error;
    protected NNMatrix[] errorNL;

    @Override
    public int[] size() {
        return new int[]{width, depth};
    }

    @Override
    public void initialize(Optimizer optimizer) {

    }

    @Override
    public int info() {
        System.out.println("Flatten\t\t|  " + width + ",\t" + depth + "\t\t|  " + countNeuron + "\t\t\t|");
        return 0;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Float to half 2D\n");
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
        output = new NNMatrix[inputs.length];

        if (Use.CPU) {
            GPU_Sleep();
            for (int i = 0; i < output.length; i++) {
                output[i] = new NNMatrix(input[i].getRow(), input[i].getColumn(), input[i].getData(), input[i].getSdata(), true);
            }
            GPU_WakeUp();
        }

        if (Use.GPU) {
            for (int i = 0; i < output.length; i++) {
                output[i] = new NNMatrix(input[i].getRow(), input[i].getColumn(), true);
                output[i].float2HalfVector(input[i]);
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
        error = new NNMatrix[errors.length];

        if (Use.CPU) {
            for (int i = 0; i < errors.length; i++) {
                GPU_Sleep();
                error[i] = new NNMatrix(errorNL[i].getRow(), errorNL[i].getColumn(), errorNL[i].getData(), errorNL[i].getSdata());
                GPU_WakeUp();
            }
        }

        if (Use.GPU) {
            for (int i = 0; i < output.length; i++) {
                error[i] = new NNMatrix(errorNL[i].getRow(), errorNL[i].getColumn());
                error[i].half2FloatVector(errorNL[i]);
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

    public static Float2Half2D read(Scanner scanner){
        return new Float2Half2D();
    }
}
