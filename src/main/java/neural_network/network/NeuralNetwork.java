package neural_network.network;

import lombok.Getter;
import neural_network.activation.FunctionActivation;
import neural_network.layers.NeuralLayer;
import neural_network.layers.dense.ActivationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.layers.dense.DenseNeuralLayer;
import neural_network.loss.FunctionLoss;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class NeuralNetwork {
    @Getter
    private ArrayList<NeuralLayer> layers;

    private int[] inputSize;
    private boolean trainable;

    private FunctionLoss functionLoss;
    private Optimizer optimizer;

    public NeuralNetwork() {
        layers = new ArrayList<>();
        trainable = true;
    }

    public NeuralNetwork addInputLayer(int... size) {
        inputSize = size;

        return this;
    }

    public NeuralNetwork create() {
        layers.get(0).initialize(inputSize);
        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).initialize(layers.get(i - 1).size());
        }

        for (NeuralLayer layer : layers) {
            layer.initialize(optimizer);
        }

        return this;
    }

    public NeuralNetwork setTrainable(boolean trainable) {
        for (NeuralLayer layer : layers) {
            layer.trainable(trainable);
        }

        return this;
    }

    public NNArray[] getOutputs() {
        return layers.get(layers.size() - 1).getOutput();
    }

    public NNArray[] getError() {
        return layers.get(0).getError();
    }

    public NeuralNetwork addLayer(NeuralLayer layer) {
        layers.add(layer);

        return this;
    }

    public NeuralNetwork setFunctionLoss(FunctionLoss functionLoss) {
        this.functionLoss = functionLoss;

        return this;
    }

    public void info() {
        int countParam = 0;
        System.out.println("\t\t\t\t NEURAL NETWORK ");
        System.out.println("==========================================================");
        System.out.println("Layer\t\t| Input size \t| Output size \t| Count param");
        for (NeuralLayer neuralLayer : layers) {
            System.out.println("____________|_______________|_______________|_____________");
            countParam += neuralLayer.info();
        }
        System.out.println("____________|_______________|_______________|_____________");
        System.out.println("       Total param:         |\t\t" + countParam);
        System.out.println("____________________________|_____________________________");
    }

    public NeuralNetwork setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
        return this;
    }

    public void save(FileWriter fileWriter) throws IOException {
        fileWriter.write("Neural network\n");
        for (int j : inputSize) {
            fileWriter.write(j + " ");
        }
        fileWriter.write("\n");
        fileWriter.flush();

        for (NeuralLayer layer : layers) {
            layer.write(fileWriter);
        }
        fileWriter.write("End\n");
        fileWriter.flush();
    }

    public static NeuralNetwork read(Scanner scanner) throws Exception {
        if (scanner.nextLine().equals("Neural network")) {
            NeuralNetwork network = new NeuralNetwork()
                    .addInputLayer(Integer.parseInt(scanner.nextLine()));
            NeuralLayer.read(scanner, network.layers);

            return network;
        }
        throw new Exception("Network is not deep");
    }

    public void queryTrain(NNArray[] input) {
        layers.get(0).generateTrainOutput(input);
        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).generateTrainOutput(layers.get(i - 1).getOutput());
        }
    }

    public NNArray[] query(NNArray[] input) {
        layers.get(0).generateOutput(input);
        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).generateOutput(layers.get(i - 1).getOutput());
        }

        return getOutputs();
    }

    public float train(NNArray[] input, NNArray[] idealOutput) {
        queryTrain(input);
        backpropagation(findDerivative(idealOutput));
        update();
        return functionLoss.findAccuracy(layers.get(layers.size() - 1).getOutput(), idealOutput);
    }

    public void forwardBackpropagation(NNArray[] input, NNArray[] idealOutput) {
        query(input);
        this.trainable = false;
        backpropagation(findDerivative(idealOutput));
        this.trainable = true;
    }

    public void train(NNArray[] error) {
        backpropagation(error);
        update();
    }

    private NNArray[] findDerivative(NNArray[] idealOutput) {
        if(layers.get(layers.size() - 1) instanceof DenseNeuralLayer){
            return NNArrays.toVector(functionLoss.findDerivative(layers.get(layers.size() - 1).getOutput(), idealOutput));
        }
        return null;
    }

    private void update() {
        optimizer.update();
        if (trainable) {
            for (NeuralLayer layer : layers) {
                layer.update(optimizer);
            }
        }
    }

    private void backpropagation(NNArray[] error) {
        layers.get(layers.size() - 1).generateError(error);
        for (int i = layers.size() - 2; i >= 0; i--) {
            layers.get(i).generateError(layers.get(i + 1).getError());
        }
    }

    public float accuracy(NNArray[] idealOutput) {
        return functionLoss.findAccuracy(getOutputs(), idealOutput);
    }

    public NeuralNetwork addDenseLayer(int countNeuron) {
        return addLayer(new DenseLayer(countNeuron));
    }

    public NeuralNetwork addDenseLayer(int countNeuron, FunctionActivation functionActivation) {
        addLayer(new DenseLayer(countNeuron));
        return addActivationLayer(functionActivation);
    }

    public NeuralNetwork addActivationLayer(FunctionActivation functionActivation) {
        return addLayer(new ActivationLayer(functionActivation));
    }

//    public NeuralNetwork addDropoutLayer(double dropout) {
//        return addLayer(new DropoutLayer(dropout));
//    }

//    public DeepNeuralNetwork addBatchNormalizationLayer(double momentum) {
//        return addLayer(new BatchNormalizationLayer(momentum));
//    }
//
//    public DeepNeuralNetwork addBatchNormalizationLayer() {
//        return addLayer(new BatchNormalizationLayer());
//    }
}
