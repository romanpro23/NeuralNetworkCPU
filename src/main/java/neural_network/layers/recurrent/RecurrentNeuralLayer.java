package neural_network.layers.recurrent;

import lombok.NoArgsConstructor;
import neural_network.initialization.Initializer;
import neural_network.layers.layer_2d.NeuralLayer2D;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNVector;

@NoArgsConstructor
public abstract class RecurrentNeuralLayer extends NeuralLayer2D {
    protected boolean returnSequences;
    protected int countNeuron;
    protected double recurrentDropout;

    protected NNVector[][] inputState;
    protected NNVector[][] state;

    protected NNVector[][] errorState;

    protected Regularization regularization;
    protected Initializer initializerInput;
    protected Initializer initializerHidden;

    protected boolean loadWeight;
    protected boolean dropout;

    public RecurrentNeuralLayer(int countNeuron, double recurrentDropout) {
        this.countNeuron = countNeuron;
        this.recurrentDropout = (float) recurrentDropout;

        this.trainable = true;
        this.initializerInput = new Initializer.XavierUniform();
        this.initializerHidden = new Initializer.XavierNormal();
        this.dropout = false;
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        generateOutput(inputs, null);
    }

    @Override
    public void generateError(NNArray[] errors) {
        generateError(errors, null);
    }

    @Override
    public void generateTrainOutput(NNArray[] inputs) {
        dropout = true;
        generateOutput(inputs);
        dropout = false;
    }

    public void generateTrainOutput(NNArray[] inputs, NNArray[][] state) {
        dropout = true;
        generateOutput(inputs, state);
        dropout = false;
    }

    public NNVector[][] getState() {
            return state;
    }

    public NNVector[][] getErrorState() {
            return errorState;
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 2) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }

        width = size[0];
        depth = size[1];
        if (returnSequences) {
            outWidth = width;
        } else {
            outWidth = 1;
        }
        outDepth = countNeuron;
    }

    public RecurrentNeuralLayer setReturnSequences(boolean returnSequences) {
        this.returnSequences = returnSequences;

        return this;
    }

    protected void copy(RecurrentNeuralLayer layer) {
        this.initializerInput = layer.initializerInput;
        this.initializerHidden = layer.initializerHidden;

        this.regularization = layer.regularization;
        this.trainable = layer.trainable;
    }

    public abstract void generateOutput(NNArray[] input, NNArray[][] state);

    public abstract void generateError(NNArray[] error, NNArray[][] errorState);
}
