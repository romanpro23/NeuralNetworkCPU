package neural_network.layers.capsule;

import neural_network.layers.doubles.DoubleNeuralLayer2D;
import neural_network.layers.layer_2d.NeuralLayer2D;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MaskLayer extends DoubleNeuralLayer2D {
    private NNVector[] mask;

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isMatrix(inputs);
        this.output = input;
    }

    @Override
    public void generateOutput(NNArray[] inputs, NNArray[] masks) {
        this.input = NNArrays.isMatrix(inputs);
        this.mask = NNArrays.isVector(masks);
        this.output = new NNMatrix[inputs.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                this.output[i] = input[i].mask(mask[i]);
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    @Override
    public NNArray[] getErrorFirst() {
        return error;
    }

    @Override
    public NNArray[] getErrorSecond() {
        return error;
    }

    @Override
    public void generateError(NNArray[] error) {
        errorNL = getErrorNextLayer(error);
        this.error = new NNMatrix[errorNL.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                this.error[i] = errorNL[i].mask(mask[i]);
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    @Override
    public int info() {
        System.out.println("Mask\t\t| " + width + ",\t" + depth + "\t\t| " + width + ",\t" + depth + "\t\t|");
        return 0;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Mask layer\n");
        writer.flush();
    }

    @Override
    public NNArray[] getErrorNL(){
        return error;
    }
    public static MaskLayer read(Scanner scanner) {
        return new MaskLayer();
    }
}