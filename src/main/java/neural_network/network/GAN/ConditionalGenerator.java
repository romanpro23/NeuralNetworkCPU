package neural_network.network.GAN;

import neural_network.layers.LayersBlock;
import neural_network.layers.NeuralLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class ConditionalGenerator {
    private NeuralNetwork generator;
    private LayersBlock noiseLayers;
    private LayersBlock labelLayers;

    private Optimizer optimizer;

    public ConditionalGenerator(){

    }

    public ConditionalGenerator setLabelLayers(LayersBlock labelLayers) {
        this.labelLayers = labelLayers;

        return this;
    }

    public ConditionalGenerator setGenerator(NeuralNetwork generator) {
        this.generator = generator;

        return this;
    }

    public ConditionalGenerator setNoiseLayers(LayersBlock noiseLayers) {
        this.noiseLayers = noiseLayers;

        return this;
    }

    public ConditionalGenerator create() {
        noiseLayers.initialize();
        labelLayers.initialize();
        generator.create();

        if (optimizer != null) {
            noiseLayers.initialize(optimizer);
            labelLayers.initialize(optimizer);
        }

        return this;
    }

    public ConditionalGenerator setTrainable(boolean trainable) {
        noiseLayers.setTrainable(trainable);
        labelLayers.setTrainable(trainable);
        generator.setTrainable(trainable);

        return this;
    }

    public NNArray[] getOutputs() {
        return generator.getOutputs();
    }

    public NNArray[] getError() {
        return generator.getError();
    }

    public ConditionalGenerator setFunctionLoss(FunctionLoss functionLoss) {
        generator.setFunctionLoss(functionLoss);

        return this;
    }

    public void info() {
        int countParam = 0;
        System.out.println("\t\t\t\t NEURAL NETWORK ");
        System.out.println("==========================================================");
        System.out.println("Layer\t\t| Input size \t| Output size \t| Count param");
        countParam += labelLayers.info();
        countParam += noiseLayers.info();
        for (NeuralLayer neuralLayer : generator.getLayers()) {
            System.out.println("____________|_______________|_______________|_____________");
            countParam += neuralLayer.info();
        }
        System.out.println("____________|_______________|_______________|_____________");
        System.out.println("       Total param:         |\t\t" + countParam);
        System.out.println("____________________________|_____________________________");
    }

    public ConditionalGenerator setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
        generator.setOptimizer(optimizer);
        return this;
    }

    public void save(FileWriter fileWriter) throws IOException {
        fileWriter.write("Neural network\n");
        labelLayers.save(fileWriter);
        noiseLayers.save(fileWriter);
        for (int j : generator.getInputSize()) {
            fileWriter.write(j + " ");
        }
        fileWriter.write("\n");
        fileWriter.flush();

        for (NeuralLayer layer : generator.getLayers()) {
            layer.save(fileWriter);
        }
        fileWriter.write("End\n");
        fileWriter.flush();
        fileWriter.close();
    }

    public static ConditionalGenerator read(Scanner scanner) throws Exception {
        if (scanner.nextLine().equals("Neural network")) {
            ConditionalGenerator generator = new ConditionalGenerator();
            generator.labelLayers = LayersBlock.read(scanner);
            generator.noiseLayers = LayersBlock.read(scanner);
            NeuralNetwork network = new NeuralNetwork()
                    .addInputLayer(Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray());
            NeuralLayer.read(scanner, network.getLayers());

            return generator;
        }
        throw new Exception("Network is not deep");
    }

    public NNArray[] queryTrain(NNArray[] noise, NNArray[] labels) {
        labelLayers.generateTrainOutput(labels);
        noiseLayers.generateTrainOutput(noise);
        generator.queryTrain(NNArrays.concat(noiseLayers.getOutput(), labelLayers.getOutput()));

        return getOutputs();
    }

    public NNArray[] query(NNArray[] noise, NNArray[] labels) {
        labelLayers.generateOutput(labels);
        noiseLayers.generateOutput(noise);
        generator.query(NNArrays.concat(noiseLayers.getOutput(), labelLayers.getOutput()));

        return getOutputs();
    }

//    public float train(NNArray[] input, NNArray[] idealOutput) {
//        return train(input, idealOutput, true);
//    }
//
//    public float train(NNArray[] input, NNArray[] idealOutput, float lambda) {
//        return train(input, idealOutput, true, lambda);
//    }
//
//    public float train(NNArray[] input, NNArray[] idealOutput, boolean update) {
//        return train(input, idealOutput, update, 1);
//    }
//
//    public float train(NNArray[] input, NNArray[] idealOutput, boolean update, float lambda) {
//        queryTrain(input);
//        backpropagation(findDerivative(idealOutput, lambda));
//        if (update) {
//            update();
//        }
//        return lambda * functionLoss.findAccuracy(layers.get(layers.size() - 1).getOutput(), idealOutput);
//    }
//
//    public float forwardBackpropagation(NNArray[] input, NNArray[] idealOutput) {
//        query(input);
//        backpropagation(findDerivative(idealOutput));
//
//        return functionLoss.findAccuracy(layers.get(layers.size() - 1).getOutput(), idealOutput);
//    }

    public void train(NNArray[] error) {
        train(error, true);
    }

    public void train(NNArray[] error, boolean update) {
        generator.train(error, false);
        labelLayers.generateError(NNArrays.subArray(generator.getError(), labelLayers.getOutput(), noiseLayers.getOutput()[0].size()));
        labelLayers.generateError(NNArrays.subArray(generator.getError(), noiseLayers.getOutput()));
        if (update) {
            update();
        }
    }

    public int[] getOutputSize() {
        return generator.getOutputSize();
    }

    public int size(){
        return generator.size() + labelLayers.getLayers().size() + noiseLayers.getLayers().size();
    }

    public void update() {
        optimizer.update();
    }

    public float accuracy(NNArray[] idealOutput) {
        return generator.accuracy(idealOutput);
    }
}
