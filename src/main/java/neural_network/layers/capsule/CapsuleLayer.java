package neural_network.layers.capsule;

import lombok.Setter;
import neural_network.initialization.Initializer;
import neural_network.layers.layer_2d.NeuralLayer2D;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CapsuleLayer extends NeuralLayer2D {
    //trainable parts
    private Regularization regularization;
    private Initializer initializer;

    private boolean loadWeight;

    //weightAttention
    @Setter
    private NNTensor4D weight;
    private NNTensor4D derWeight;

    private final int countRouting;
    private final int sizeVector;
    private final int countCapsule;

    private NNTensor[] transformInput;
    private NNMatrix[][] inputSquash;
    private NNMatrix[][] outputSquash;
    private NNMatrix[][] c;

    public CapsuleLayer(int countCapsule, int sizeVector) {
        this(countCapsule, sizeVector, 3);
    }

    public CapsuleLayer(int countCapsule, int sizeVector, int countRouting) {
        this.countCapsule = countCapsule;
        this.countRouting = countRouting;
        this.sizeVector = sizeVector;
        trainable = true;

        initializer = new Initializer.RandomNormal();
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 2) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        this.depth = size[1];
        this.width = size[0];

        outWidth = countCapsule;
        outDepth = sizeVector;

        derWeight = new NNTensor4D(countCapsule, width, sizeVector, depth);
        if (!loadWeight) {
            weight = new NNTensor4D(countCapsule, width, sizeVector, depth);
            initializer.initialize(weight);
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(weight, derWeight);
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isMatrix(inputs);
        output = new NNMatrix[inputs.length];
        transformInput = new NNTensor[inputs.length];
        inputSquash = new NNMatrix[inputs.length][countRouting];
        outputSquash = new NNMatrix[inputs.length][countRouting];
        c = new NNMatrix[inputs.length][countRouting];

        ExecutorService executor = Executors.newFixedThreadPool(inputs.length);
        for (int t = 0; t < inputs.length; t++) {
            final int i = t;
            executor.execute(() -> {
                output[i] = new NNMatrix(outWidth, outDepth);
                transformInput[i] = input[i].dot(weight);
                NNMatrix b = new NNMatrix(countCapsule, width);

                for (int j = 0; j < countRouting; j++) {
                    c[i][j] = new NNMatrix(countCapsule, width);
                    outputSquash[i][j] = new NNMatrix(countCapsule, sizeVector);
                    c[i][j].softmax(b);

                    inputSquash[i][j] = c[i][j].dot(transformInput[i]);
                    outputSquash[i][j].squash(inputSquash[i][j]);

                    if (j < countRouting - 1) {
                        b.addScalarMul(transformInput[i], outputSquash[i][j]);
                    } else {
                        output[i] = outputSquash[i][j];
                    }
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        error = new NNMatrix[errors.length];

        ExecutorService executor = Executors.newFixedThreadPool(input.length);
        for (int t = 0; t < input.length; t++) {
            final int i = t;
            executor.execute(() -> {
                NNMatrix errorSquash = new NNMatrix(outWidth, outDepth);
                errorSquash.derSquash(inputSquash[i][countRouting - 1], errorNL[i]);

                NNTensor errorTransformInput = c[i][countRouting - 1].dotR(errorSquash);
                NNMatrix errorC = errorSquash.dotT(transformInput[i]);
                NNMatrix errorB = new NNMatrix(countCapsule, width);
                errorB.derSoftmax(c[i][countRouting - 1], errorC);

                for (int j = countRouting - 2; j >= 0; j--) {
                    errorTransformInput.addDerScalarMul(outputSquash[i][j], errorB);

                    NNMatrix deltaSquash = c[i][j].derScalarMul(transformInput[i], errorB);
                    errorSquash.derSquash(inputSquash[i][j], deltaSquash);
                    errorTransformInput.add(c[i][j].dotR(errorSquash));

                    if(j > 0) {
                        errorC = errorSquash.dotT(transformInput[i]);
                        NNMatrix deltaB = new NNMatrix(countCapsule, width);
                        deltaB.derSoftmax(c[i][j], errorC);
                        errorB.add(deltaB);
                    }
                }

                error[i] = weight.dot(errorTransformInput);
                if (trainable) {
                    derWeight.addMatrixDot(input[i], errorTransformInput);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        if (trainable && regularization != null) {
            regularization.regularization(weight);
        }
    }

    @Override
    public int info() {
        int countParam = weight.size();
        System.out.println("Capsule\t\t| " + width + ",\t" + depth + "\t\t| " + outWidth + ",\t" + outDepth + "\t\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Capsule layer\n");
        writer.write(countCapsule + " " + sizeVector + " " + countRouting + "\n");
        weight.save(writer);
        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    public static CapsuleLayer read(Scanner scanner) {
        int[] param = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();

        CapsuleLayer layer = new CapsuleLayer(param[0], param[1], param[2]);
        layer.loadWeight = false;
        layer.weight = NNTensor4D.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }

    public CapsuleLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;
        return this;
    }

    public CapsuleLayer setTrainable(boolean trainable) {
        this.trainable = trainable;
        return this;
    }

    public CapsuleLayer setInitializer(Initializer initializer) {
        this.initializer = initializer;
        return this;
    }

    public void setLoadWeight(boolean loadWeight) {
        this.loadWeight = loadWeight;
    }
}
