package neural_network.layers.layer_2d;

import lombok.Getter;
import lombok.Setter;
import neural_network.initialization.Initializer;
import neural_network.layers.reshape.EmbeddingLayer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class VITPositionalEmbeddingLayer extends NeuralLayer2D {
    //trainable parts
    private Regularization regularization;
    @Setter
    private boolean loadWeight;

    @Getter
    private NNMatrix weight;
    private NNMatrix derWeight;

    public VITPositionalEmbeddingLayer() {
        trainable = true;
    }

    @Override
    public void initialize(int[] size) {
        super.initialize(size);

        derWeight = new NNMatrix(width, depth);
        if (!loadWeight) {
            weight = new NNMatrix(width, depth);
        }
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isMatrix(input);
        this.output = new NNMatrix[input.length];

        for (int i = 0; i < output.length; i++) {
            this.output[i] = new NNMatrix(this.input[i]);
            this.output[i].copy(this.input[i]);
            this.output[i].add(weight);
        }
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        if(trainable){
            for (int i = 0; i < errors.length; i++) {
                derWeight.add(errorNL[i]);
            }

            if(regularization != null){
                regularization.regularization(weight);
            }
        }
        this.error = errorNL;
    }


    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(weight, derWeight);
    }

    @Override
    public int info() {
        System.out.println("Position emb| " + width + ",\t" + depth + "\t\t| " + outWidth + ",\t" + outDepth + "\t\t|");
        return 0;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("VIT positional embedding layer\n");
        weight.save(writer);
        if(regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    public static VITPositionalEmbeddingLayer read(Scanner scanner) {
        VITPositionalEmbeddingLayer layer = new VITPositionalEmbeddingLayer();
        layer.weight = NNMatrix.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }

    public VITPositionalEmbeddingLayer setRegularization(Regularization regularization){
        this.regularization = regularization;
        return this;
    }

    public VITPositionalEmbeddingLayer setTrainable(boolean trainable){
        this.trainable = trainable;
        return this;
    }
}
