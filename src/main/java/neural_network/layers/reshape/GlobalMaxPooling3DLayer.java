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

public class GlobalMaxPooling3DLayer extends Flatten3DLayer {

    @Override
    public int info() {
        System.out.println("Global max  |  " + height + ",\t" + width + ",\t" + depth + "\t|  " + countNeuron + "\t\t\t|");
        return 0;
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Global max pooling 3D\n");
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
        countNeuron = depth;
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        input = NNArrays.isTensor(inputs);
        output = new NNVector[inputs.length];

        for (int i = 0; i < output.length; i++) {
            output[i] = new NNVector(countNeuron);
            output[i].globalMaxPool(input[i]);
        }
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        error = new NNTensor[errors.length];

        for (int i = 0; i < errors.length; i++) {
            error[i] = new NNTensor(height, width, depth);
            error[i].backGlobalMaxPool(input[i], output[i], errorNL[i]);
        }
    }

    public static GlobalMaxPooling3DLayer read(Scanner scanner){
        return new GlobalMaxPooling3DLayer();
    }
}
