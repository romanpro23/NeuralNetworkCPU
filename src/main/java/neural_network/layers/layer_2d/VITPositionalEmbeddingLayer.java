package neural_network.layers.layer_2d;

import jcuda.driver.JCudaDriver;
import lombok.Getter;
import lombok.Setter;
import neural_network.initialization.Initializer;
import neural_network.layers.reshape.EmbeddingLayer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import utilities.Use;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static utilities.JCudaHelper.CONTEXT;
import static utilities.Use.*;

public class VITPositionalEmbeddingLayer extends NeuralLayer2D {
    //trainable parts
    private Regularization regularization;
    @Setter
    private boolean loadWeight;

    @Getter
    private NNMatrix weight;
    private NNMatrix derWeight;

    public VITPositionalEmbeddingLayer(boolean half) {
        trainable = true;
        this.half = half;
    }

    @Override
    public void initialize(int[] size) {
        super.initialize(size);

        derWeight = new NNMatrix(width, depth, half);
        if (!loadWeight) {
            weight = new NNMatrix(width, depth, half);
        }
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isMatrix(input);
        this.output = new NNMatrix[input.length];

        if ((Use.CPU) && (!Use.GPU)) {
            GPU_Sleep();
            ExecutorService executor = Executors.newFixedThreadPool(output.length);
            for (int t = 0; t < output.length; t++) {
                final int i = t;
                executor.execute(() -> {
                    this.output[i] = new NNMatrix(this.input[i], half);
                    this.output[i].copy(this.input[i]);
                    this.output[i].add(weight);
                });
            }
            executor.shutdown();
            while (!executor.isTerminated()) {
            }
            GPU_WakeUp();
        }

        if (Use.GPU) {
            for (int i = 0; i < output.length; i++) {
                this.output[i] = new NNMatrix(this.input[i], half);
                this.output[i].copy(this.input[i]);
                this.output[i].add(weight);
            }
        }
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        if(trainable){
            if ((Use.CPU) && (!Use.GPU)) {
                GPU_Sleep();
                ExecutorService executor = Executors.newFixedThreadPool(errors.length);
                for (int t = 0; t < errors.length; t++) {
                    final int i = t;
                    executor.execute(() -> {
                        derWeight.add(errorNL[i]);
                    });
                }
                executor.shutdown();
                while (!executor.isTerminated()) {
                }
                GPU_WakeUp();
            }
            if (Use.GPU) {
                for (int i = 0; i < errors.length; i++) {
                    derWeight.add(errorNL[i]);
                }
            }

            if (regularization != null) {
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
        writer.write(this.half + "\n");
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
        VITPositionalEmbeddingLayer layer = new VITPositionalEmbeddingLayer(Boolean.parseBoolean(scanner.nextLine()));
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
