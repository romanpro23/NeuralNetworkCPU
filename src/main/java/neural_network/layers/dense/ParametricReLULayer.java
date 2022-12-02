package neural_network.layers.dense;

import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.convolution_3d.ParametricReLULayer3D;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class ParametricReLULayer extends DenseNeuralLayer {
    private Regularization regularization;
    private Initializer initializer;

    private boolean loadWeight;

    private NNVector alpha;
    private NNVector derAlpha;

    public ParametricReLULayer() {
        trainable = true;

        initializer = null;
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 1) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }
        countNeuron = size[0];

        derAlpha = new NNVector(countNeuron);
        if (!loadWeight) {
            alpha = new NNVector(countNeuron);
            if (initializer != null) {
                initializer.initialize(alpha);
            }
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(alpha, derAlpha);
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isVector(input);
        this.output = new NNVector[input.length];

        for (int i = 0; i < output.length; i++) {
            this.output[i] = new NNVector(countNeuron);
            output[i].prelu(this.input[i], alpha);
        }
    }

    @Override
    public void generateError(NNArray[] error) {
        errorNL = getErrorNextLayer(error);
        this.error = new NNVector[errorNL.length];

        for (int i = 0; i < input.length; i++) {
            this.error[i] = new NNVector(countNeuron);
            this.error[i].derPrelu(input[i], errorNL[i], alpha);
            if(trainable){
                derivativeWeight(input[i], errorNL[i]);
            }
        }

        if (trainable && regularization != null){
            regularization.regularization(alpha);
        }
    }

    private void derivativeWeight(NNVector input, NNVector error) {
        for (int i = 0, index = 0; i < countNeuron; i++) {
            if (input.get(index) < 0) {
                derAlpha.getData()[i] += input.get(index) * error.get(index);
            }
        }
    }

    @Override
    public int info() {
        System.out.println("Activation\t|  " + countNeuron + "\t\t\t|  " + countNeuron + "\t\t\t|");
        return 0;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Parametric ReLU activation layer\n");
        alpha.save(writer);
        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    public ParametricReLULayer setRegularization(Regularization regularization){
        this.regularization = regularization;
        return this;
    }

    public ParametricReLULayer setTrainable(boolean trainable){
        this.trainable = trainable;
        return this;
    }

    public static ParametricReLULayer read(Scanner scanner) {
        ParametricReLULayer layer = new ParametricReLULayer();
        layer.loadWeight = false;
        layer.alpha = NNVector.read(scanner);
        layer.setRegularization(Regularization.read(scanner));
        layer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        layer.loadWeight = true;
        return layer;
    }
}