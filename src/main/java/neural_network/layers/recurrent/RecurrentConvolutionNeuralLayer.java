package neural_network.layers.recurrent;

import lombok.NoArgsConstructor;
import neural_network.initialization.Initializer;
import neural_network.layers.layer_4d.ConvolutionNeuralLayer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNTensor;

@NoArgsConstructor
public abstract class RecurrentConvolutionNeuralLayer extends ConvolutionNeuralLayer {
    protected boolean returnSequences;

    protected int paddingY;
    protected int paddingX;
    protected int step;
    protected int heightKernel;
    protected int widthKernel;
    protected int countKernel;
    protected double recurrentDropout;

    protected NNTensor[][] inputHidden;
    protected NNTensor[][] outputHidden;
    protected NNTensor[][] state;
    protected NNTensor[][] errorState;

    protected Regularization regularization;
    protected Initializer initializerInput;
    protected Initializer initializerHidden;

    protected boolean loadWeight;
    protected boolean dropout;

    public RecurrentConvolutionNeuralLayer(int countKernel, int heightKernel, int widthKernel, int step, int paddingY, int paddingX, double recurrentDropout) {
        this.countKernel = countKernel;
        this.paddingX = paddingX;
        this.paddingY = paddingY;
        this.step = step;
        this.heightKernel = heightKernel;
        this.widthKernel = widthKernel;

        this.recurrentDropout = (float) recurrentDropout;

        this.trainable = true;
        this.initializerInput = new Initializer.XavierUniform();
        this.initializerHidden = new Initializer.XavierNormal();
        this.dropout = false;
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 4) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }

        length = size[0];
        height = size[1];
        width = size[2];
        depth = size[3];
        if (returnSequences) {
            outLength = length;
        } else {
            outLength = 1;
        }
        outWidth = (width - widthKernel + 2 * paddingX) / step + 1;
        outHeight = (height - heightKernel + 2 * paddingY) / step + 1;
        outDepth = countKernel;
    }

    protected void copy(RecurrentConvolutionNeuralLayer layer){
        this.initializerInput = layer.initializerInput;
        this.initializerHidden = layer.initializerHidden;
        this.regularization = layer.regularization;
        this.trainable = layer.trainable;

        this.countKernel = layer.countKernel;
        this.paddingX = layer.paddingX;
        this.paddingY = layer.paddingY;
        this.step = layer.step;
        this.heightKernel = layer.heightKernel;
        this.widthKernel = layer.widthKernel;
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

    public NNTensor[][] getState() {
        return state;
    }

    public NNTensor[][] getErrorState() {
        return errorState;
    }

    public abstract void generateOutput(NNArray[] input, NNArray[][] state);

    public abstract void generateError(NNArray[] error, NNArray[][] errorState);
}
