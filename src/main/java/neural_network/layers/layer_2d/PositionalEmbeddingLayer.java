package neural_network.layers.layer_2d;

import lombok.Getter;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class PositionalEmbeddingLayer extends NeuralLayer2D {
    @Getter
    private NNMatrix positionalVal;

    @Override
    public void initialize(int[] size) {
        super.initialize(size);

        positionalVal = new NNMatrix(size[0], size[1]);

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < depth / 2; j ++) {
                positionalVal.set(i, 2*j, (float) Math.sin(i / Math.pow(10000, (2.0 * j) / depth)));
                positionalVal.set(i, 2*j + 1, (float) Math.cos(i / Math.pow(10000, (2.0 * j + 1) / depth)));
            }
        }
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isMatrix(input);
        this.output = new NNMatrix[input.length];

        for (int i = 0; i < output.length; i++) {
            this.output[i] = new NNMatrix(this.input[i]);
            this.output[i].copy(this.input[i]);
            this.output[i].add(positionalVal);
        }
    }

    @Override
    public void generateError(NNArray[] error) {
        errorNL = getErrorNextLayer(error);
        this.error = errorNL;
    }

    @Override
    public int info() {
        System.out.println("Position emb| " + width + ",\t" + depth + "\t\t| " + outWidth + ",\t" + outDepth + "\t\t|");
        return 0;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Positional embedding layer\n");
        writer.flush();
    }

    public static PositionalEmbeddingLayer read(Scanner scanner) {
        return new PositionalEmbeddingLayer();
    }
}
