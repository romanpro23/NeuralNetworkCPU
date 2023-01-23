package neural_network.layers.layer_3d.attention;

import neural_network.activation.FunctionActivation;
import neural_network.layers.LayersBlock;
import neural_network.layers.NeuralLayer;
import neural_network.layers.layer_1d.*;
import neural_network.layers.reshape.Flatten3DLayer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class SEBlock extends LayersBlock {
    private int depth, height, width;
    private int outHeight, outWidth, outDepth;

    private NNTensor[] input;
    private NNTensor[] output;
    private NNVector[] outputBlock;

    private NNTensor[] error;
    private NNVector[] errorBlock;

    public SEBlock() {
        layers = new ArrayList<>();
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 3) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }

        height = size[0];
        width = size[1];
        depth = size[2];
        outWidth = width;
        outHeight = height;
        outDepth = depth;

        super.initialize(size);
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("SE block\n");
        for (NeuralLayer neuralLayer : layers) {
            neuralLayer.save(writer);
        }
        writer.write("End\n");
        writer.flush();
    }

    public static SEBlock read(Scanner scanner) {
        SEBlock seBlock = new SEBlock();
        NeuralLayer.read(scanner, seBlock.layers);

        return seBlock;
    }

    @Override
    public void generateOutput(NNArray[] inputs){
        this.input = NNArrays.isTensor(inputs);
        super.generateOutput(inputs);
        generateOutput();
    }

    private void generateOutput(){
        this.output = new NNTensor[input.length];
        this.outputBlock = NNArrays.isVector(super.getOutput());
        for (int i = 0; i < input.length; i++) {
            this.output[i] = input[i].mul(outputBlock[i]);
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] inputs){
        this.input = NNArrays.isTensor(inputs);
        super.generateTrainOutput(inputs);
        generateOutput();
    }

    private void generateErrors(NNTensor[] errors){
        this.error = new NNTensor[input.length];
        this.errorBlock = new NNVector[input.length];
        for (int i = 0; i < input.length; i++) {
            this.error[i] = errors[i].mul(outputBlock[i]);
            this.errorBlock[i] = errors[i].mul(input[i]);
        }
    }

    public void addErrors(){
        for (int i = 0; i < error.length; i++) {
            error[i].add(super.getError()[i]);
        }
    }

    @Override
    public void generateError(NNArray[] errors){
        generateErrors(NNArrays.isTensor(errors));
        super.generateError(errorBlock);
        addErrors();
    }

    @Override
    public NNArray[] getOutput(){
        return output;
    }

    @Override
    public NNArray[] getError(){
        return error;
    }

    public SEBlock addGlobalPoolingLayer(Flatten3DLayer layer) {
        layers.add(layer);

        return this;
    }

    public SEBlock addDenseLayer(DenseLayer denseLayer) {
        layers.add(denseLayer);
        return this;
    }

    public SEBlock addDenseLayer(int countNeurons) {
        DenseLayer denseLayer = new DenseLayer(countNeurons);
        addDenseLayer(denseLayer);
        return this;
    }

    public SEBlock addDenseLayer(int countNeurons, FunctionActivation functionActivation) {
        DenseLayer denseLayer = new DenseLayer(countNeurons);
        addDenseLayer(denseLayer);
        addActivationLayer(functionActivation);
        return this;
    }

    public SEBlock addActivationLayer(FunctionActivation functionActivation) {
        addActivationLayer(new ActivationLayer(functionActivation));
        return this;
    }

    public SEBlock addActivationLayer(ActivationLayer activationLayer) {
        layers.add(activationLayer);
        return this;
    }

    public SEBlock addDropoutLayer(double chanceDropNeuron) {
        DropoutLayer dropoutLayer = new DropoutLayer(chanceDropNeuron);
        layers.add(dropoutLayer);
        return this;
    }

    public SEBlock addDropoutLayer(DropoutLayer dropoutLayer) {
        layers.add(dropoutLayer);
        return this;
    }

    public SEBlock addBatchNormalizationLayer(BatchNormalizationLayer batchNormalizationLayer) {
        layers.add(batchNormalizationLayer);
        return this;
    }

    public SEBlock addBatchNormalizationLayer() {
        return addBatchNormalizationLayer(new BatchNormalizationLayer());
    }

    public SEBlock addLayer(NeuralLayer layer) {
        layers.add(layer);
        return this;
    }

    @Override
    public int info() {
        int countParam = 0;
        System.out.println("            | Squeeze-and-Excitation layers |             ");
        System.out.println("____________|_______________________________|_____________");
        for (NeuralLayer neuralLayer : layers) {
            countParam += neuralLayer.info();
            System.out.println("____________|_______________|_______________|_____________");
        }
        System.out.println("            |  " + height + ",\t" + width + ",\t" + depth + "\t|  "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        System.out.println("____________|_______________|_______________|_____________");
        return countParam;
    }

    public SEBlock setTrainable(boolean trainable) {
        super.setTrainable(trainable);

        return this;
    }

    @Override
    public int[] size() {
        return new int[]{outHeight, outWidth, outDepth};
    }
}
