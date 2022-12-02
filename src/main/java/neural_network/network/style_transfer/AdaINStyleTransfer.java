package neural_network.network.style_transfer;

import lombok.NoArgsConstructor;
import neural_network.layers.dual.AdaINLayer3D;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;

@NoArgsConstructor
public class AdaINStyleTransfer {
    private NeuralNetwork encoder;
    private NeuralNetwork decoder;
    private AdaINLayer3D adaINLayer;

    private NNTensor[] content;
    private NNTensor[] style;
    private NNTensor[] input;

    private float accuracy;

    private FunctionLoss loss;

    public AdaINStyleTransfer(NeuralNetwork encoder, NeuralNetwork decoder) {
        this.encoder = encoder;
        this.decoder = decoder;
        this.adaINLayer = new AdaINLayer3D();

        this.loss = new FunctionLoss.MSE();
    }

    public AdaINStyleTransfer setContent(NNTensor content) {
        this.content = NNArrays.isTensor(encoder.query(new NNTensor[]{content}));

        return this;
    }

    public AdaINStyleTransfer setStyle(NNTensor style) {
        this.style = NNArrays.isTensor(encoder.query(new NNTensor[]{style}));
        return this;
    }

    public NNTensor[] getResult() {
        return NNArrays.isTensor(decoder.getOutputs());
    }

    public AdaINStyleTransfer create() {
        encoder.setTrainable(false);
        this.adaINLayer.initialize(encoder.getOutputSize());

        if (content != null && style != null) {
            adaINLayer.generateOutput(content, style);
            this.input = NNArrays.isTensor(adaINLayer.getOutput());
        }

        return this;
    }

    public float train() {
        return train(1.0, 1.0);
    }

    public float train(double styleVal, double contentVal) {
        accuracy = 0;

        decoder.queryTrain(input);
        encoder.queryTrain(decoder.getOutputs());
        encoder.train(generateError(styleVal, contentVal), false);
        decoder.train(encoder.getError());

        return accuracy;
    }

    private NNTensor[] generateError(double styleVal, double contentVal) {
        accuracy += loss.findAccuracy(encoder.getOutputs(), input);
        NNTensor[] errorContent = NNArrays.toTensor(loss.findDerivative(encoder.getOutputs(), input), input[0].shape());
        NNTensor[] errorStyle = errorStyle();
        for (int i = 0; i < errorContent.length; i++) {
            errorContent[i].mul((float) contentVal);
//            errorContent[i].mul((float) 0);
//            errorStyle[i].mul((float) styleVal);
//            errorContent[i].add(errorStyle[i]);
        }

        return errorContent;
    }

    private NNTensor[] errorStyle() {
        NNTensor[] error = new NNTensor[encoder.getOutputs().length];
        for (int i = 0; i < encoder.getOutputs().length; i++) {
            NNVector mean = findMean((NNTensor) encoder.getOutputs()[i]);
            NNVector var = findVariance((NNTensor) encoder.getOutputs()[i], mean);

            accuracy += loss.findAccuracy(mean, adaINLayer.getMeanSecond()[i]);
            accuracy += loss.findAccuracy(var, adaINLayer.getVarSecond()[i]);

            NNVector derMean = NNArrays.toVector(loss.findDerivative(mean, adaINLayer.getMeanSecond()[i]))[0];
            NNVector derVar = NNArrays.toVector(loss.findDerivative(var, adaINLayer.getVarSecond()[i]))[0];

            error[i] = findError((NNTensor) encoder.getOutputs()[i], derMean, derVar, mean, var);
        }
        return error;
    }

    private NNTensor findError(NNTensor input, NNVector derMean, NNVector derVar, NNVector mean, NNVector var) {
        NNTensor error = new NNTensor(input.shape());
        int depth = input.getDepth();
        int size = input.getColumns() * input.getRows();
        float epsilon = 0.0000001f;

        derMean.div(size);
        derVar.mul(2.0f / (size));

        float[] dVar = new float[var.size()];
        for (int i = 0; i < var.size(); i++) {
            dVar[i] = (float) (1.0 / Math.sqrt(var.getData()[i] + epsilon));
        }

        int index = 0;
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < depth; k++, index++) {
                error.getData()[index] = derVar.get(k) * (input.get(index) - mean.get(k)) + derMean.get(k);
            }
        }

        return error;
    }

    public NNVector findMean(NNTensor input) {
        int depth = input.getDepth();
        int size = input.getColumns() * input.getRows();

        NNVector mean = new NNVector(depth);

        int index = 0;
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < depth; k++, index++) {
                mean.getData()[k] += input.getData()[index];
            }
        }
        mean.div(size);

        return mean;
    }

    public NNVector findVariance(NNTensor input, NNVector mean) {
        int depth = input.getDepth();
        int size = input.getColumns() * input.getRows();

        NNVector var = new NNVector(depth);
        float sub;
        int index = 0;
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < depth; k++, index++) {
                sub = input.getData()[index] - mean.getData()[k];
                var.getData()[k] += sub * sub;
            }
        }
        var.div(size);

        return var;
    }
}
