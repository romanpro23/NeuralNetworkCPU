package neural_network.layers.reshape;

import nnarrays.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class GlobalMaxPooling2DLayer extends Flatten2DLayer {

    @Override
    public int info() {
        System.out.println("Global max  |  " + width + ",\t" + depth + "\t\t|  " + countNeuron + "\t\t\t|");
        return 0;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Global max pooling 2D\n");
        writer.flush();
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 2) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        depth = size[1];
        width = size[0];
        countNeuron = depth;
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        input = NNArrays.isMatrix(inputs);
        output = new NNVector[inputs.length];

        for (int i = 0; i < output.length; i++) {
            output[i] = new NNVector(countNeuron);
            output[i].globalMaxPool(input[i]);
        }
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        error = new NNMatrix[errors.length];

        for (int i = 0; i < errors.length; i++) {
            error[i] = new NNMatrix(width, depth);
            error[i].backGlobalMaxPool(input[i], output[i], errorNL[i]);
        }
    }

    public static GlobalMaxPooling2DLayer read(Scanner scanner){
        return new GlobalMaxPooling2DLayer();
    }
}
