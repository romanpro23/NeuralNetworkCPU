package neural_network.layers.layer_2d;

import neural_network.layers.NeuralLayer;
import neural_network.layers.layer_3d.NeuralLayer3D;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNTensor;
import utilities.Use;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

import static utilities.Use.GPU_Sleep;
import static utilities.Use.GPU_WakeUp;

public class TYPE2Float3D extends NeuralLayer3D {
    protected int height, width, depth;
    protected int countNeuron;
    protected NNTensor[] input;
    protected NNTensor[] output;
    protected NNTensor[] error;
    protected NNTensor[] errorNL;

    @Override
    public int[] size() {
        return new int[]{width, height, depth};
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
        writer.write("TYPE to float 3D\n");
        writer.flush();
    }

    @Override
    public void initialize(int[] size) {
        //if (size.length != 2) {
        //    throw new ExceptionInInitializerError("Error size pre layer!");
        //}

        height = size[0];
        width = size[1];
        depth = size[2];

        countNeuron = depth * width * height;
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        input = NNArrays.isTensor(inputs);
        output = new NNTensor[inputs.length];

        if ((Use.CPU) && (!Use.GPU)) {
            GPU_Sleep();
            for (int i = 0; i < output.length; i++) {
                output[i] = new NNTensor(input[i].getRows(), input[i].getColumns(), input[i].getDepth(), input[i].getData(), input[i].getSdata());
            }
            GPU_WakeUp();
        }

        if (Use.GPU) {
            for (int i = 0; i < output.length; i++) {
                output[i] = new NNTensor(input[i].getRows(), input[i].getColumns(), input[i].getDepth());
                output[i].TYPE2FloatVector(input[i]);
            }
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        generateOutput(input);
    }

    @Override
    public void generateError(NNArray[] errors) {
        /*errorNL = getErrorNextLayer(errors);
        error = new NNTensor[errors.length];

        for (int i = 0; i < errors.length; i++) {
            if (Use.CPU) {
                GPU_Sleep();
                error[i] = new NNTensor(errorNL[i].getRows(), errorNL[i].getColumns(), errorNL[i].getDepth(), errorNL[i].getData(), errorNL[i].getSdata(), true);
                GPU_WakeUp();
            }

            if (Use.GPU) {
                error[i] = new NNTensor(errorNL[i].getRows(), errorNL[i].getColumns(), errorNL[i].getDepth(),true);
                error[i].float2TYPEVector(errorNL[i]);
            }
        }*/
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
    public NNArray[] getError() {
        return error;
    }

    public static TYPE2Float3D read(Scanner scanner){
        return new TYPE2Float3D();
    }
}
