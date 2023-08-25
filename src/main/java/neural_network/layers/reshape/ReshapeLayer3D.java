package neural_network.layers.reshape;

import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;
import utilities.CublasUtil;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class ReshapeLayer3D extends NeuralLayer {
    private final int depth;
    private final int height;
    private final int width;
    protected int countNeuron;

    protected NNTensor[] output;
    protected NNVector[] input;
    protected NNTensor[] errorNL;
    protected NNVector[] error;

    public ReshapeLayer3D(int height, int width, int depth) {
        this.depth = depth;
        this.height = height;
        this.width = width;
    }

    @Override
    public int[] size() {
        return new int[]{height, width, depth};
    }

    @Override
    public void initialize(Optimizer optimizer) {
        //no have initialize elements
    }

    @Override
    public int info() {
        System.out.println("Reshape\t\t|  " + countNeuron + "\t\t\t|  " + height + ",\t" + width + ",\t" + depth + "\t|");
        return 0;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Reshape layer 3D\n");
        writer.write(height + " " + width + " " + depth + "\n");
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
        output = new NNTensor[inputs.length];

        for (int i = 0; i < output.length; i++) {
            output[i] = new NNTensor(height, width, depth, input[i].getData());
        }
    }

    @Override
    public void generateOutput(CublasUtil.Matrix[] input_gpu) {

    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        generateOutput(input);
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        error = new NNVector[errors.length];

        for (int i = 0; i < errors.length; i++) {
            error[i] = new NNVector(errorNL[i].getData());
        }
    }

    public NNTensor[] getErrorNextLayer(NNArray[] error) {
        NNTensor[] errorNL = NNArrays.isTensor(error);

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
    public CublasUtil.Matrix[] getOutput_gpu() {
        return new CublasUtil.Matrix[0];
    }

    @Override
    public NNArray[] getError() {
        return error;
    }

    @Override
    public CublasUtil.Matrix[] getError_gpu() {
        return new CublasUtil.Matrix[0];
    }

    public static ReshapeLayer3D read(Scanner scanner) {
        int[] param = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
        return new ReshapeLayer3D(param[0], param[1], param[2]);
    }
}
