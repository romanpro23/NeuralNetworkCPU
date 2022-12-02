package neural_network.network;

import lombok.Getter;
import neural_network.activation.FunctionActivation;
import neural_network.layers.NeuralLayer;
import neural_network.layers.convolution_3d.ActivationLayer3D;
import neural_network.layers.convolution_3d.BatchNormalizationLayer3D;
import neural_network.layers.convolution_3d.ConvolutionNeuralLayer;
import neural_network.layers.convolution_3d.DropoutLayer3D;
import neural_network.layers.convolution_3d.u_net.ConcatenateLayer;
import neural_network.layers.dense.*;
import neural_network.loss.FunctionLoss;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class NeuralNetwork {
    @Getter
    private final ArrayList<NeuralLayer> layers;

    private int[] inputSize;
    private int stopGradient;

    private FunctionLoss functionLoss;
    @Getter
    private Optimizer optimizer;

    public NeuralNetwork() {
        layers = new ArrayList<>();
        stopGradient = 0;
    }

    public NeuralNetwork copy() {
        NeuralNetwork newNetwork = new NeuralNetwork()
                .addInputLayer(inputSize);
        newNetwork.layers.addAll(this.layers);
        newNetwork.functionLoss = functionLoss;
        newNetwork.optimizer = optimizer;

        return newNetwork;
    }
    
    public NeuralNetwork removeLastLayers(int count){
        for (int i = layers.size() - 1, c = 0; c < count; i--, c++) {
            layers.remove(i);
        }

        return this;
    }

    public NeuralLayer getLastLayer(){
        return layers.get(layers.size() - 1);
    }

    public NeuralNetwork setStopGradient(int stopGradient) {
        this.stopGradient = stopGradient;

        return this;
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

        if (optimizer != null) {
            for (int i = stopGradient; i < layers.size(); i++) {
                if (layers.get(i).isTrainable()) {
                    layers.get(i).initialize(optimizer);
                }
            }
        }

        return this;
    }

    public NeuralNetwork initialize(Optimizer optimizer) {
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

    public NeuralNetwork addConcatenateLayer(int index) {
        layers.add(new ConcatenateLayer().addLayer(layers.get(index), index));

        return this;
    }

    public NeuralNetwork addLayers(ArrayList<NeuralLayer> layers) {
        this.layers.addAll(layers);

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
            layer.save(fileWriter);
        }
        fileWriter.write("End\n");
        fileWriter.flush();
    }

    public static NeuralNetwork read(Scanner scanner) throws Exception {
        if (scanner.nextLine().equals("Neural network")) {
            NeuralNetwork network = new NeuralNetwork()
                    .addInputLayer(Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray());
            NeuralLayer.read(scanner, network.layers);

            return network;
        }
        throw new Exception("Network is not deep");
    }

    public NNArray[] queryTrain(NNArray[] input) {
        layers.get(0).generateTrainOutput(input);
        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).generateTrainOutput(layers.get(i - 1).getOutput());
        }

        return getOutputs();
    }

    public NNArray[] query(NNArray[] input) {
        layers.get(0).generateOutput(input);
        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).generateOutput(layers.get(i - 1).getOutput());
        }

        return getOutputs();
    }

    public float train(NNArray[] input, NNArray[] idealOutput) {
        return train(input, idealOutput, true);
    }

    public float train(NNArray[] input, NNArray[] idealOutput, float lambda) {
        return train(input, idealOutput, true, lambda);
    }

    public float train(NNArray[] input, NNArray[] idealOutput, boolean update) {
        return train(input, idealOutput, update, 1);
    }

    public float train(NNArray[] input, NNArray[] idealOutput, boolean update, float lambda) {
        queryTrain(input);
        backpropagation(findDerivative(idealOutput, lambda));
        if (update) {
            update();
        }
        return lambda * functionLoss.findAccuracy(layers.get(layers.size() - 1).getOutput(), idealOutput);
    }

    public float forwardBackpropagation(NNArray[] input, NNArray[] idealOutput) {
        query(input);
        backpropagation(findDerivative(idealOutput));

        return functionLoss.findAccuracy(layers.get(layers.size() - 1).getOutput(), idealOutput);
    }

    public void train(NNArray[] error) {
        train(error, true);
    }

    public void train(NNArray[] error, boolean update) {
        backpropagation(error);
        if (update) {
            update();
        }
    }

    public int[] getInputSize() {
        return inputSize;
    }

    public int[] getOutputSize() {
        return layers.get(layers.size() - 1).size();
    }

    public NNArray[] findDerivative(NNArray[] idealOutput) {
        return findDerivative(idealOutput, 1);
    }

    public NNArray[] findDerivative(NNArray[] idealOutput, float lambda) {
        int[] size = layers.get(layers.size() - 1).size();
        NNArray[] result = null;
        if (size.length == 1) {
            result = NNArrays.toVector(functionLoss.findDerivative(layers.get(layers.size() - 1).getOutput(), idealOutput));
        } else if (size.length == 2) {
            result = NNArrays.toMatrix(functionLoss.findDerivative(layers.get(layers.size() - 1).getOutput(), idealOutput),
                    size[0], size[1]);
        } else if (size.length == 3) {
            result = NNArrays.toTensor(functionLoss.findDerivative(layers.get(layers.size() - 1).getOutput(), idealOutput),
                    size[0], size[1], size[2]);
        }

        if (lambda != 1) {
            for (NNArray array : result) {
                array.mul(lambda);
            }
        }

        return result;
    }

    public void update() {
        optimizer.update();
    }

    private void backpropagation(NNArray[] error) {
        layers.get(layers.size() - 1).generateError(error);
        for (int i = layers.size() - 2; i >= stopGradient; i--) {
            layers.get(i).generateError(layers.get(i + 1).getError());
        }
    }

    public float accuracy(NNArray[] idealOutput) {
        return functionLoss.findAccuracy(getOutputs(), idealOutput);
    }

    public ArrayList<NeuralLayer> getConvolutionLayers() {
        ArrayList<NeuralLayer> convLayers = new ArrayList<>();
        for (NeuralLayer layer : layers) {
            if (layer instanceof ConvolutionNeuralLayer) {
                convLayers.add(layer);
            } else {
                break;
            }
        }

        return convLayers;
    }

    public ArrayList<NeuralLayer> getLayers(int first, int last) {
        ArrayList<NeuralLayer> neuralLayers = new ArrayList<>();
        for (int i = first; i < last; i++) {
            neuralLayers.add(layers.get(i));
        }

        return neuralLayers;
    }

    public NeuralNetwork addDenseLayer(int countNeuron) {
        return addLayer(new DenseLayer(countNeuron));
    }

    public NeuralNetwork addDenseLayer(int countNeuron, FunctionActivation functionActivation) {
        addLayer(new DenseLayer(countNeuron));
        return addActivationLayer(functionActivation);
    }

    public NeuralNetwork addActivationLayer(FunctionActivation functionActivation) {
        if (layers.get(layers.size() - 1).size().length == 3) {
            return addLayer(new ActivationLayer3D(functionActivation));
        }
        return addLayer(new ActivationLayer(functionActivation));
    }

    public NeuralNetwork addDropoutLayer(double dropout) {
        if (layers.get(layers.size() - 1).size().length == 3) {
            return addLayer(new DropoutLayer3D(dropout));
        }
        return addLayer(new DropoutLayer(dropout));
    }

    public NeuralNetwork addBatchNormalizationLayer(double momentum) {
        if (layers.get(layers.size() - 1).size().length == 3) {
            return addLayer(new BatchNormalizationLayer3D(momentum));
        }
        return addLayer(new BatchNormalizationLayer(momentum));
    }

    public NeuralNetwork addBatchNormalizationLayer() {
        return addBatchNormalizationLayer(0.99);
    }
}
