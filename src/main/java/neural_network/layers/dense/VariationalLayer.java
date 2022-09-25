package neural_network.layers.dense;

import lombok.Setter;
import neural_network.initialization.Initializer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class VariationalLayer extends DenseNeuralLayer {
    private DenseLayer mu;
    private DenseLayer gamma;

    private Initializer randomInitializer;

    private boolean trainable;
    private boolean randomVariational;

    private NNVector[] errorMu, errorGamma, random;

    private VariationalLayer() {
        this.trainable = true;
        this.randomVariational = true;
        this.randomInitializer = new Initializer.RandomNormal();
    }

    public VariationalLayer(int countNeuron) {
        this();
        this.countNeuron = countNeuron;
        mu = new DenseLayer(countNeuron);
        gamma = new DenseLayer(countNeuron);
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 1) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        mu.initialize(size);
        gamma.initialize(size);
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isVector(inputs);
        output = new NNVector[input.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = new NNVector(countNeuron);
        }

        if (randomVariational) {
            random = new NNVector[input.length];
            for (int i = 0; i < random.length; i++) {
                random[i] = new NNVector(countNeuron);
                randomInitializer.initialize(random[i]);
            }
            gamma.generateOutput(input);

            NNVector[] gammaOutput = NNArrays.isVector(gamma.getOutput());
            for (int i = 0; i < gammaOutput.length; i++) {
                output[i].addProduct(random[i], gammaOutput[i]);
            }
        }

        mu.generateOutput(input);
        NNVector[] muOutput = NNArrays.isVector(mu.getOutput());
        for (int i = 0; i < output.length; i++) {
            output[i].add(muOutput[i]);
        }
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNVector[errorNL.length];

        gamma.setTrainable(trainable);
        mu.setTrainable(trainable);

        generateErrorMu();
        NNVector[] errorMu = NNArrays.isVector(mu.getError());
        NNVector[] errorGamma = null;

        if (randomVariational) {
            generateErrorGamma();
            errorGamma = NNArrays.isVector(gamma.getError());
        }

        for (int i = 0; i < error.length; i++) {
            error[i] = new NNVector(input[i].size());
            error[i].add(errorMu[i]);
            if (randomVariational && errorGamma != null) {
                error[i].add(errorGamma[i]);
            }
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        if (optimizer.getCountParam() > 0) {
            mu.initialize(optimizer);
            gamma.initialize(optimizer);
        }
    }

    @Override
    public void update(Optimizer optimizer) {
        if (trainable) {
            mu.update(optimizer);
            gamma.update(optimizer);
        }
    }

    private void generateErrorMu() {
        errorMu = new NNVector[errorNL.length];
        NNVector[] muOutput = NNArrays.isVector(mu.getOutput());

        for (int i = 0; i < errorMu.length; i++) {
            errorMu[i] = new NNVector(countNeuron);
            errorMu[i].add(errorNL[i]);
            errorMu[i].add(muOutput[i]);
        }

        mu.generateError(errorMu);
    }

    public void generateErrorGamma() {
        errorGamma = new NNVector[errorNL.length];
        NNVector[] gammaOutput = NNArrays.isVector(gamma.getOutput());

        for (int i = 0; i < errorGamma.length; i++) {
            errorGamma[i] = new NNVector(countNeuron);
            errorGamma[i].add(gammaOutput[i]);
            errorGamma[i].addProduct(errorNL[i], random[i]);
            errorGamma[i].subOneDiv(gammaOutput[i]);
        }

        gamma.generateError(errorGamma);
    }

    public VariationalLayer setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public VariationalLayer setInitializer(Initializer initializer) {
        gamma.setInitializer(initializer);
        mu.setInitializer(initializer);

        return this;
    }

    public VariationalLayer setRandomInitializer(Initializer initializer) {
        this.randomInitializer = initializer;

        return this;
    }

    public VariationalLayer setRandomVariational(boolean randomVariational) {
        this.randomVariational = randomVariational;

        return this;
    }

    public VariationalLayer setRegularization(Regularization regularization) {
        mu.setRegularization(regularization);
        gamma.setRegularization(regularization);

        return this;
    }

    public float findKLDivergence(){
        float KLD = 0;
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < countNeuron; j++) {
                KLD += Math.pow(mu.getOutput()[i].get(j), 2) + Math.pow(gamma.getOutput()[i].get(j), 2)
                    - 1 - Math.log(Math.pow(gamma.getOutput()[i].get(j), 2));
            }
        }
        return KLD;
    }

    @Override
    public int info() {
        System.out.println("            |       Variational layer       |             ");
        System.out.println("____________|_______________|_______________|_____________");
        int countParam = mu.info();
        System.out.println("____________|_______________|_______________|_____________");
        countParam += gamma.info();
        System.out.println("____________|_______________|_______________|_____________");
        System.out.println("            |  " + gamma.getWeight().getColumn() + "\t\t\t|  " + countNeuron + "\t\t\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Variational layer\n");
        mu.write(writer);
        gamma.write(writer);
        writer.write(trainable + "\n");
        writer.flush();
    }

    public static VariationalLayer read(Scanner scanner) {
        VariationalLayer layer = new VariationalLayer();
        //read mu and gamma
        scanner.nextLine();
        layer.mu = DenseLayer.read(scanner);
        scanner.nextLine();
        layer.gamma = DenseLayer.read(scanner);
        layer.countNeuron = layer.gamma.countNeuron;

        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        return layer;
    }
}
