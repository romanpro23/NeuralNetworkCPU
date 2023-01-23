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

    private NNMatrix[] c;

    private final int countRouting;
    private final int sizeVector;
    private final int countCapsule;

    private NNTensor[] transformInput;
    private NNMatrix[] inputSquash;

    public CapsuleLayer(int countCapsule, int sizeVector) {
        this(countCapsule, sizeVector, 3);
    }

    public CapsuleLayer(int countCapsule, int sizeVector, int countRouting) {
        this.countCapsule = countCapsule;
        this.countRouting = countRouting;
        this.sizeVector = sizeVector;
        trainable = true;

        initializer = new Initializer.HeNormal();
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

        derWeight = new NNTensor4D(countCapsule, width, depth, sizeVector);
        if (!loadWeight) {
            weight = new NNTensor4D(countCapsule, width, depth, sizeVector);
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
        inputSquash = new NNMatrix[inputs.length];
        c = new NNMatrix[inputs.length];

        ExecutorService executor = Executors.newFixedThreadPool(inputs.length);
        for (int t = 0; t < inputs.length; t++) {
            final int i = t;
            executor.execute(() -> {
                output[i] = new NNMatrix(outWidth, outDepth);
                transformInput[i] = input[i].capsuleAffineTransform(weight);
                NNMatrix bias = new NNMatrix(countCapsule, width);
                c[i] = new NNMatrix(countCapsule, width);
                for (int j = 0; j < countRouting; j++) {
                    c[i].softmax(bias);
                    inputSquash[i] = transformInput[i].weightSum(c[i]);
                    output[i].squash(inputSquash[i]);
                    if (j != countRouting - 1) {
                        bias.addScalarMul(transformInput[i], output[i]);
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
                errorSquash.derSquash(inputSquash[i], errorNL[i]);
                NNTensor errorTransformInput = transformInput[i].backWeightSum(c[i], errorSquash);
                error[i] = input[i].derCapsuleAffineTransform(weight, errorTransformInput);

                if (trainable) {
                    derWeight.derCapsuleAffineTransform(input[i], errorTransformInput);
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
