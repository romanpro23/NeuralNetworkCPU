package neural_network.layers.reshape;

import lombok.Getter;
import lombok.Setter;
import neural_network.initialization.Initializer;
import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class EmbeddingLayer extends NeuralLayer {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;
    private boolean trainable;
    @Setter
    private boolean loadWeight;

    protected NNVector[] input;
    protected NNMatrix[] output;
    protected NNMatrix[] errorNL;

    private int sizeInput;
    private final int sizeVocabulary;
    @Getter
    private final int lengthVector;

    private NNMatrix weight;
    private NNMatrix derWeight;

    public EmbeddingLayer(int sizeVocabulary, int lengthVector) {
        this.sizeVocabulary = sizeVocabulary;
        this.lengthVector = lengthVector;
        initializer = new Initializer.RandomUniform();
        trainable = true;
    }

    @Override
    public void initialize(int[] size){
        if (size.length != 1) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        sizeInput = size[0];
        derWeight = new NNMatrix(sizeVocabulary, lengthVector);

        if(!loadWeight){
            weight = new NNMatrix(sizeVocabulary, lengthVector);
            initializer.initialize(weight);
        }
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isVector(input);
        this.output = new NNMatrix[input.length];

        int indexWord, index, indexOutput;
        for (int i = 0; i < output.length; i++) {
            this.output[i] = new NNMatrix(this.input[i].size(), lengthVector);
            indexOutput = 0;
            for (int j = 0; j < this.input[i].size(); j++) {
                indexWord = (int) this.input[i].get(j);
                index = indexWord * lengthVector;
                for (int k = 0; k < lengthVector; k++, index++, indexOutput++) {
                    this.output[i].getData()[indexOutput] = weight.getData()[index];
                }
            }
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] input) {
        generateOutput(input);
    }

    @Override
    public void generateError(NNArray[] error) {
        if(trainable) {
            errorNL = NNArrays.isMatrix(error);
            int indexWord, index, indexOutput;

            for (int i = 0; i < output.length; i++) {
                indexOutput = 0;
                for (int j = 0; j < this.input[i].size(); j++) {
                    indexWord = (int) this.input[i].get(j);
                    index = indexWord * lengthVector;
                    for (int k = 0; k < lengthVector; k++, index++, indexOutput++) {
                        derWeight.getData()[index] += errorNL[i].getData()[indexOutput];
                    }
                }
            }

            if (input.length != 1) {
                derWeight.div(input.length);
            }
            if (regularization != null) {
                regularization.regularization(weight);
            }
        }
    }

    @Override
    public int[] size() {
        return new int[]{sizeInput, lengthVector};
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(weight, derWeight);
    }

    @Override
    public int info() {
        int countParam = weight.size();
        System.out.println("Embedding\t|  " + sizeInput + "\t\t\t|  " + sizeInput + ",\t" + lengthVector + "\t\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Embedding layer\n");
        writer.write(sizeVocabulary + " " + lengthVector  + "\n");
        weight.save(writer);
        if(regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    @Override
    public NNArray[] getOutput() {
        return output;
    }

    @Override
    public NNArray[] getError() {
        return errorNL;
    }

    public static EmbeddingLayer read(Scanner scanner){
        int[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
        EmbeddingLayer layer = new EmbeddingLayer(arr[0], arr[1]);
        layer.weight = NNMatrix.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }

    public EmbeddingLayer setRegularization(Regularization regularization){
        this.regularization = regularization;
        return this;
    }

    public EmbeddingLayer setTrainable(boolean trainable){
        this.trainable = trainable;
        return this;
    }

    public EmbeddingLayer setInitializer(Initializer initializer){
        this.initializer = initializer;
        return this;
    }
}
