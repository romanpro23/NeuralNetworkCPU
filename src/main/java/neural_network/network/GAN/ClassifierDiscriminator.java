package neural_network.network.GAN;

import lombok.NoArgsConstructor;
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

@NoArgsConstructor
public class ClassifierDiscriminator {
    private NeuralNetwork discriminator;
    private LayersBlock classifierLayers;
    private LayersBlock discriminatorLayers;

    private Optimizer optimizer;
    private FunctionLoss ganLoss;
    private FunctionLoss classifierLoss;

    public ClassifierDiscriminator(NeuralNetwork network){
        this.discriminator = network;
    }

    public ClassifierDiscriminator setDiscriminatorLayers(LayersBlock discriminatorLayers) {
        this.discriminatorLayers = discriminatorLayers;

        return this;
    }

    public ClassifierDiscriminator setDiscriminator(NeuralNetwork discriminator) {
        this.discriminator = discriminator;

        return this;
    }

    public ClassifierDiscriminator setClassifierLayers(LayersBlock classifierLayers) {
        this.classifierLayers = classifierLayers;

        return this;
    }

    public ClassifierDiscriminator create() {
        discriminator.create();
        classifierLayers.initialize(discriminator.getOutputSize());
        discriminatorLayers.initialize(discriminator.getOutputSize());

        if (optimizer != null) {
            classifierLayers.initialize(optimizer);
            discriminatorLayers.initialize(optimizer);
        }

        return this;
    }

    public ClassifierDiscriminator setTrainable(boolean trainable) {
        classifierLayers.setTrainable(trainable);
        discriminatorLayers.setTrainable(trainable);
        discriminator.setTrainable(trainable);

        return this;
    }

    public NNArray[] getOutputs() {
        return classifierLayers.getOutput();
    }

    public NNArray[] getError() {
        return discriminator.getError();
    }

    public ClassifierDiscriminator setFunctionLoss(FunctionLoss functionLoss) {
        discriminator.setFunctionLoss(functionLoss);

        return this;
    }

    public ClassifierDiscriminator setGANFunctionLoss(FunctionLoss functionLoss) {
        ganLoss = functionLoss;

        return this;
    }

    public ClassifierDiscriminator setClassifierFunctionLoss(FunctionLoss functionLoss) {
        classifierLoss = functionLoss;

        return this;
    }

    public void info() {
        int countParam = 0;
        System.out.println("\t\t\t\t NEURAL NETWORK ");
        System.out.println("==========================================================");
        System.out.println("Layer\t\t| Input size \t| Output size \t| Count param");
        for (NeuralLayer neuralLayer : discriminator.getLayers()) {
            System.out.println("____________|_______________|_______________|_____________");
            countParam += neuralLayer.info();
        }
        System.out.println("____________|_______________|_______________|_____________");
        countParam += discriminatorLayers.info();
        countParam += classifierLayers.info();
        System.out.println("       Total param:         |\t\t" + countParam);
        System.out.println("____________________________|_____________________________");
    }

    public ClassifierDiscriminator setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
        discriminator.setOptimizer(optimizer);
        return this;
    }

    public void save(FileWriter fileWriter) throws IOException {
        fileWriter.write("Neural network\n");

        for (int j : discriminator.getInputSize()) {
            fileWriter.write(j + " ");
        }
        fileWriter.write("\n");
        fileWriter.flush();

        for (NeuralLayer layer : discriminator.getLayers()) {
            layer.save(fileWriter);
        }
        fileWriter.write("End\n");
        discriminatorLayers.save(fileWriter);
        classifierLayers.save(fileWriter);
        fileWriter.flush();
        fileWriter.close();
    }

    public static ClassifierDiscriminator read(Scanner scanner) throws Exception {
        if (scanner.nextLine().equals("Neural network")) {
            ClassifierDiscriminator generator = new ClassifierDiscriminator();

            NeuralNetwork network = new NeuralNetwork()
                    .addInputLayer(Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray());
            NeuralLayer.read(scanner, network.getLayers());
            generator.discriminator = network;
            generator.discriminatorLayers = LayersBlock.read(scanner);
            generator.classifierLayers = LayersBlock.read(scanner);

            return generator;
        }
        throw new Exception("Network is not deep");
    }

    public NNArray[] queryTrain(NNArray[] inputs) {
        discriminator.queryTrain(inputs);
        discriminatorLayers.generateTrainOutput(discriminator.getOutputs());
        classifierLayers.generateTrainOutput(discriminator.getOutputs());

        return getOutputs();
    }

    public NNArray[] query(NNArray[] inputs) {
        discriminator.query(inputs);
        discriminatorLayers.generateOutput(discriminator.getOutputs());
        classifierLayers.generateOutput(discriminator.getOutputs());

        return getOutputs();
    }

    public float train(NNArray[] input, NNArray[] labels, NNArray[] output) {
        return train(input, labels, output, true,1);
    }

    public float train(NNArray[] input, NNArray[] labels, NNArray[] output, boolean update, float lambda) {
        queryTrain(input);
        backpropagation(labels, output, lambda);
        if (update) {
            update();
        }
        return accuracy(labels, output);
    }

    private void backpropagation(NNArray[] labels, NNArray[] output, float lambda){
        discriminatorLayers.generateError(findDerivative(ganLoss, discriminatorLayers.getOutput(), output, 1));
        classifierLayers.generateError(findDerivative(classifierLoss, classifierLayers.getOutput(), labels, lambda));

        NNArrays.add(discriminatorLayers.getError(), classifierLayers.getError());
        discriminator.train(discriminatorLayers.getError(), false);
    }

    private float accuracy(NNArray[] labels, NNArray[] output){
        float accuracy = ganLoss.findAccuracy(discriminatorLayers.getOutput(), output);
        accuracy += classifierLoss.findAccuracy(classifierLayers.getOutput(), labels);
        return accuracy;
    }

    public NNArray[] findDerivative(FunctionLoss functionLoss ,NNArray[] output, NNArray[] idealOutput, float lambda) {
        int[] size = idealOutput[0].shape();
        NNArray[] result = null;
        if (size.length == 1) {
            result = NNArrays.toVector(classifierLoss.findDerivative(output, idealOutput));
        } else if (size.length == 2) {
            result = NNArrays.toMatrix(functionLoss.findDerivative(output, idealOutput),
                    size[0], size[1]);
        } else if (size.length == 3) {
            result = NNArrays.toTensor(functionLoss.findDerivative(output, idealOutput),
                    size[0], size[1], size[2]);
        }

        if (lambda != 1) {
            for (NNArray array : result) {
                array.mul(lambda);
            }
        }

        return result;
    }

    public int[] getOutputSize() {
        return discriminatorLayers.size();
    }

    public int size(){
        return discriminator.size() + discriminatorLayers.getLayers().size() + classifierLayers.getLayers().size();
    }

    public void update() {
        optimizer.update();
    }
}
