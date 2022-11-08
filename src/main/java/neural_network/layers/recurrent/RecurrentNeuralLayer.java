package neural_network.layers.recurrent;

import lombok.NoArgsConstructor;
import neural_network.initialization.Initializer;
import neural_network.layers.convolution_2d.ConvolutionNeuralLayer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNVector;

@NoArgsConstructor
public abstract class RecurrentNeuralLayer extends ConvolutionNeuralLayer {
    protected boolean returnSequences;
    protected boolean returnState;

    protected RecurrentNeuralLayer preLayer, nextLayer;

    protected int countNeuron;
    protected double recurrentDropout;

    protected NNVector[][] inputHidden;
    protected NNVector[][] outputHidden;
    protected NNVector[][] state;
    protected NNVector[][] errorState;

    protected NNVector[] hiddenError;

    protected Regularization regularization;
    protected Initializer initializerInput;
    protected Initializer initializerHidden;

    protected boolean loadWeight;
    protected boolean dropout;

    public RecurrentNeuralLayer(int countNeuron, double recurrentDropout) {
        this.countNeuron = countNeuron;
        this.recurrentDropout = (float) recurrentDropout;
        this.returnState = false;

        this.trainable = true;
        this.initializerInput = new Initializer.XavierUniform();
        this.initializerHidden = new Initializer.XavierNormal();
        this.dropout = false;
        preLayer = null;
    }

    @Override
    public void generateTrainOutput(NNArray[] inputs) {
        dropout = true;
        generateOutput(inputs);
        dropout = false;
    }

    public RecurrentNeuralLayer setPreLayer(RecurrentNeuralLayer layer) {
        this.preLayer = layer;
        layer.returnState = true;
        layer.nextLayer = this;

        return this;
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

    protected boolean hasPreLayer() {
        return preLayer != null;
    }

    protected void copy(RecurrentNeuralLayer layer){
        this.initializerInput = layer.initializerInput;
        this.initializerHidden = layer.initializerHidden;
        this.returnState = layer.returnState;
        this.regularization = layer.regularization;
        this.trainable = layer.trainable;
    }
}
