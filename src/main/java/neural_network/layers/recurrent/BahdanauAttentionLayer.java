package neural_network.layers.recurrent;

import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNMatrix;
import nnarrays.NNVector;

public abstract class BahdanauAttentionLayer extends RecurrentNeuralLayer {
    protected NeuralLayer encoderLayer;

    protected NNMatrix weightAttention;
    protected NNMatrix derWeightAttention;

    protected NNMatrix weightVal;
    protected NNMatrix derWeightVal;

    protected NNMatrix weightKey;
    protected NNMatrix derWeightKey;

    protected final int sizeAttention;

    protected NNMatrix[] errorInput;
    protected NNMatrix[] valInput;
    protected NNVector[][] keyInput;

    protected NNMatrix[][] inputScore;
    protected NNMatrix[][] outputScore;
    protected NNMatrix[][] inputAttention;
    protected NNMatrix[][] outputAttention;

    protected NNVector[][] contextVector;
    protected NNVector[][] inputVector;
    protected NNVector[][] hiddenInput;

    protected NNMatrix[] outputPreLayer;

    public BahdanauAttentionLayer(int countNeuron, int sizeAttention, double recurrentDropout) {
        super(countNeuron, recurrentDropout);
        this.sizeAttention = sizeAttention;
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(weightAttention, derWeightAttention);
        optimizer.addDataOptimize(weightKey, derWeightKey);
        optimizer.addDataOptimize(weightVal, derWeightVal);
    }

    protected NNVector generateAttention(int t, int i, NNVector hidden_t) {
        if (hidden_t != null) {
            hiddenInput[i][t] = hidden_t;
            keyInput[i][t] = hidden_t.dot(weightKey);
        } else {
            keyInput[i][t] = new NNVector(sizeAttention);
        }

        inputScore[i][t] = valInput[i].sum(keyInput[i][t]);
        outputScore[i][t] = new NNMatrix(inputScore[i][t]);
        outputScore[i][t].tanh(inputScore[i][t]);

        inputAttention[i][t] = outputScore[i][t].dot(weightAttention);
        outputAttention[i][t] = new NNMatrix(inputAttention[i][t].getColumn(), inputAttention[i][t].getRow());
        outputAttention[i][t].softmax(inputAttention[i][t]);

        contextVector[i][t] = new NNVector(outputAttention[i][t].dot(outputPreLayer[i]).getData());

        return contextVector[i][t];
    }

    protected NNVector generateErrorAttention(NNVector error, int i, int t){
        errorInput[i].add(outputAttention[i][t].transpose().dot(error));
        NNVector errorAtt = error.dotT(outputPreLayer[i]);
        NNMatrix errorInputAtt = new NNMatrix(inputAttention[i][t]);
        errorInputAtt.derSoftmax(outputAttention[i][t], errorAtt);

        NNMatrix errorScoreOut = errorInputAtt.dotT(weightAttention);
        NNMatrix errorScoreIn = new NNMatrix(inputScore[i][t]);
        errorScoreIn.derTanh(outputScore[i][t], errorScoreOut);
        errorInput[i].add(errorScoreIn.dotT(weightVal));

        if(trainable){
            derWeightAttention.add(outputScore[i][t].transpose().dot(errorInputAtt));
            derWeightVal.add(outputPreLayer[i].transpose().dot(errorScoreIn));
            derWeightKey.add(hiddenInput[i][t].dot(errorScoreIn.sum()));
        }

        return errorScoreIn.sum().dotT(weightKey);
    }

    protected void initializeMemory(int size) {
        this.hiddenInput = new NNVector[size][];
        this.errorInput = new NNMatrix[size];
        this.valInput = new NNMatrix[size];
        this.keyInput = new NNVector[size][];
        inputScore = new NNMatrix[size][];
        outputScore = new NNMatrix[size][];
        outputAttention = new NNMatrix[size][];
        inputAttention = new NNMatrix[size][];
        contextVector = new NNVector[size][];
        inputVector = new NNVector[size][];
    }

    protected void initializeMemory(int i, int size) {
        keyInput[i] = new NNVector[size];
        contextVector[i] = new NNVector[size];
        inputAttention[i] = new NNMatrix[size];
        outputAttention[i] = new NNMatrix[size];
        outputScore[i] = new NNMatrix[size];
        inputScore[i] = new NNMatrix[size];
        inputVector[i] = new NNVector[size];
        this.hiddenInput[i] = new NNVector[size];

        errorInput[i] = new NNMatrix(outputPreLayer[i]);
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 2) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }

        width = size[0];
        depth = size[1] + encoderLayer.size()[1];
        if (returnSequences) {
            outWidth = width;
        } else {
            outWidth = 1;
        }
        outDepth = countNeuron;

        derWeightAttention = new NNMatrix(sizeAttention, 1);
        derWeightKey = new NNMatrix(sizeAttention, countNeuron);
        derWeightVal = new NNMatrix(sizeAttention, countNeuron);

        if (!loadWeight) {
            weightAttention = new NNMatrix(sizeAttention, 1);
            weightKey = new NNMatrix(sizeAttention, countNeuron);
            weightVal = new NNMatrix(countNeuron, sizeAttention);
        }
    }

    @Override
    public NNArray[] getErrorNL() {
        return errorInput;
    }

    public BahdanauAttentionLayer setEncoderLayer(NeuralLayer layer){
        layer.addNextLayer(this);
        this.encoderLayer = layer;

        return this;
    }
}
