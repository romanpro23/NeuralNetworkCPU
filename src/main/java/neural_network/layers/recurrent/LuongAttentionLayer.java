package neural_network.layers.recurrent;

import lombok.SneakyThrows;
import neural_network.layers.NeuralLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNMatrix;
import nnarrays.NNVector;

import java.io.Writer;

public abstract class LuongAttentionLayer extends RecurrentNeuralLayer {
    protected NeuralLayer encoderLayer;

    protected NNMatrix weightAttention;
    protected NNMatrix derWeightAttention;

    protected NNMatrix weightVal;
    protected NNMatrix derWeightVal;

    protected NNMatrix[] errorInput;
    protected NNMatrix[][] concatInput;
    protected NNMatrix[][] concatOutput;

    protected NNMatrix[][] inputScore;
    protected NNVector[][] outputScore;
    protected NNMatrix[][] inputAttention;
    protected NNMatrix[][] outputAttention;

    protected NNVector[][] contextVector;
    protected NNVector[][] outputVector;
    protected NNVector[][] hiddenInput;

    protected NNMatrix[] outputPreLayer;

    protected Attention attention;

    public LuongAttentionLayer(int countNeuron, double recurrentDropout) {
        this(countNeuron, recurrentDropout, Attention.DOT);
    }

    public LuongAttentionLayer(int countNeuron, double recurrentDropout, Attention attention) {
        super(countNeuron, recurrentDropout);
        this.attention = attention;
    }

    @SneakyThrows
    public void writeAttention(Writer writer) {
        if (attention == Attention.GENERAL) {
            writer.write("GENERAL\n");
        } else if (attention == Attention.CONCAT) {
            writer.write("CONCAT\n");
        } else if (attention == Attention.DOT) {
            writer.write("DOT\n");
        }
    }

    public static Attention readAttention(String att) {
        if (att.equals("GENERAL")) {
            return Attention.GENERAL;
        } else if (att.equals("CONCAT")) {
            return Attention.CONCAT;
        }
        return Attention.DOT;
    }

    public static enum Attention {
        DOT,
        GENERAL,
        CONCAT
    }

    @Override
    public void initialize(Optimizer optimizer) {
        if (attention == Attention.GENERAL) {
            optimizer.addDataOptimize(weightVal, derWeightVal);
        } else if (attention == Attention.CONCAT) {
            optimizer.addDataOptimize(weightAttention, derWeightAttention);
            optimizer.addDataOptimize(weightVal, derWeightVal);
        }
    }

    protected NNVector generateAttention(int t, int i, NNVector hidden_t) {
        hiddenInput[i][t] = hidden_t;
        if (attention == Attention.DOT) {
            inputAttention[i][t] = outputPreLayer[i].dotT(hidden_t);
        } else if (attention == Attention.GENERAL) {
            outputScore[i][t] = hidden_t.dot(weightVal);
            inputAttention[i][t] = outputPreLayer[i].dotT(outputScore[i][t]);
        } else if (attention == Attention.CONCAT) {
            concatInput[i][t] = outputPreLayer[i].sum(hidden_t);
            inputScore[i][t] = concatInput[i][t].dot(weightVal);
            concatOutput[i][t] = new NNMatrix(inputScore[i][t]);
            concatOutput[i][t].tanh(inputScore[i][t]);
            inputAttention[i][t] = concatOutput[i][t].dot(weightAttention);
        }

        outputAttention[i][t] = new NNMatrix(inputAttention[i][t].getColumn(), inputAttention[i][t].getRow());
        outputAttention[i][t].softmax(inputAttention[i][t]);

        contextVector[i][t] = new NNVector(outputAttention[i][t].dot(outputPreLayer[i]).getData());

        return contextVector[i][t];
    }

    protected NNVector generateErrorAttention(NNVector error, int i, int t) {
        errorInput[i].add(outputAttention[i][t].transpose().dot(error));
        NNVector errorAtt = error.dotT(outputPreLayer[i]);
        NNMatrix errorInputAtt = new NNMatrix(inputAttention[i][t]);
        errorInputAtt.derSoftmax(outputAttention[i][t], errorAtt);

        NNVector errorHidden;
        if (attention == Attention.DOT) {
            errorHidden = new NNVector(outputPreLayer[i].transpose().dot(errorInputAtt));
            errorInput[i].add(errorInputAtt.dot(hiddenInput[i][t]));
        } else if (attention == Attention.GENERAL) {
            errorInput[i].add(errorInputAtt.dot(outputScore[i][t]));
            NNVector errorOutputScore = new NNVector(outputPreLayer[i].transpose().dot(errorInputAtt));
            errorHidden = errorOutputScore.dotT(weightVal);
            if(trainable){
                derWeightVal.add(hiddenInput[i][t].dot(errorOutputScore));
            }
        } else {
            NNMatrix errorOutputConcat = errorInputAtt.dotT(weightAttention);
            NNMatrix errorInputScore = new NNMatrix(inputScore[i][t]);
            errorInputScore.derTanh(concatOutput[i][t], errorOutputConcat);
            NNMatrix errorInputConcat = errorInputScore.dotT(weightVal);

            errorHidden = errorInputConcat.sum();
            errorInput[i].add(errorInputConcat);

            if(trainable){
                derWeightAttention.add(concatOutput[i][t].transpose().dot(errorInputAtt));
                derWeightVal.add(concatInput[i][t].transpose().dot(errorInputScore));
            }
        }

        return errorHidden;
    }

    protected void initializeMemory(int size) {
        this.hiddenInput = new NNVector[size][];
        this.errorInput = new NNMatrix[size];
        this.concatInput = new NNMatrix[size][];
        this.concatOutput = new NNMatrix[size][];
        inputScore = new NNMatrix[size][];
        outputScore = new NNVector[size][];
        outputAttention = new NNMatrix[size][];
        inputAttention = new NNMatrix[size][];
        contextVector = new NNVector[size][];
        outputVector = new NNVector[size][];
    }

    protected void initializeMemory(int i, int size) {
        concatInput[i] = new NNMatrix[size];
        concatOutput[i] = new NNMatrix[size];
        contextVector[i] = new NNVector[size];
        inputAttention[i] = new NNMatrix[size];
        outputAttention[i] = new NNMatrix[size];
        outputScore[i] = new NNVector[size];
        inputScore[i] = new NNMatrix[size];
        outputVector[i] = new NNVector[size];
        hiddenInput[i] = new NNVector[size];

        errorInput[i] = new NNMatrix(outputPreLayer[i]);
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
        outDepth = countNeuron * 2;

        if (attention == Attention.GENERAL) {
            derWeightVal = new NNMatrix(countNeuron, countNeuron);
        } else if (attention == Attention.CONCAT) {
            derWeightAttention = new NNMatrix(countNeuron, 1);
            derWeightVal = new NNMatrix(countNeuron, countNeuron);
        }

        if (!loadWeight) {
            if (attention == Attention.GENERAL) {
                weightVal = new NNMatrix(countNeuron, countNeuron);
            } else if (attention == Attention.CONCAT) {
                weightAttention = new NNMatrix(countNeuron, 1);
                weightVal = new NNMatrix(countNeuron, countNeuron);
            }
        }
    }

    @Override
    public NNArray[] getErrorNL() {
        return errorInput;
    }

    public LuongAttentionLayer setEncoderLayer(NeuralLayer layer){
        layer.addNextLayer(this);
        this.encoderLayer = layer;

        return this;
    }
}
