package neural_network.network.style_transfer;

import lombok.NoArgsConstructor;
import neural_network.layers.NeuralLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNTensor;

import java.util.ArrayList;
import java.util.Collections;

@NoArgsConstructor
public class StyleTransfer {
    private NeuralNetwork network;

    private ArrayList<NeuralLayer> styleLayers;
    private ArrayList<NNMatrix> styleGrammar;
    private NNTensor contentNetwork;

    private NNTensor result;
    private NNTensor delta;
    private float accuracy;

    private FunctionLoss loss;
    private Optimizer optimizer;

    public NNMatrix grammarMatrix(NNMatrix input) {
        return input.transpose().dot(input);
    }

    public NNMatrix derGrammarMatrix(NNMatrix input, NNMatrix error) {
        NNMatrix result = input.dot(error);
        result.mul(2);
        return result;
    }

    public StyleTransfer(NeuralNetwork network) {
        this.network = network;
        styleLayers = new ArrayList<>();
        styleGrammar = new ArrayList<>();

        this.loss = new FunctionLoss.MSE(8);
    }

    public StyleTransfer addStyleLayer(NeuralLayer... neuralLayers) {
        Collections.addAll(this.styleLayers, neuralLayers);

        return this;
    }

    public StyleTransfer setContent(NNTensor content) {
        contentNetwork = NNArrays.isTensor(network.query(new NNTensor[]{content}))[0];
        result = new NNTensor(content.shape());
        delta = new NNTensor(content.shape());
        this.result.add(content);

        return this;
    }

    public StyleTransfer setStyle(NNTensor style) {
        network.query(new NNTensor[]{style});

        int layerNumber = 0;
        for (int i = 0; i < network.getLayers().size(); i++) {
            if (layerNumber < styleLayers.size() && network.getLayers().get(i) == styleLayers.get(layerNumber)) {
                int[] size = styleLayers.get(layerNumber).size();
                NNMatrix style_result = new NNMatrix(size[1] * size[0], size[2], styleLayers.get(layerNumber).getOutput()[0].getData());

                styleGrammar.add(grammarMatrix(style_result));
                layerNumber++;
            }
        }

        return this;
    }

    public StyleTransfer setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;

        return this;
    }

    public StyleTransfer create(){
        optimizer.addDataOptimize(result, delta);
        return this;
    }

    public NNTensor getResult() {
        return result;
    }

    public float train(double styleVal, double contentVal) {
        accuracy = 0;
        network.query(new NNTensor[]{result});

        accuracy += loss.findAccuracy(network.getOutputs(), new NNTensor[]{contentNetwork}) * contentVal;
        NNTensor errorContent = NNArrays.toTensor(loss.findDerivative(network.getOutputs(), new NNTensor[]{contentNetwork}), contentNetwork.shape())[0];
        errorContent.mul((float) contentVal);

        ArrayList<NeuralLayer> layers = network.getLayers();
        int numberLayer = styleLayers.size() - 1;

        if (layers.get(layers.size() - 1) == styleLayers.get(numberLayer)) {
            errorContent.add(errorGrammar(styleLayers.get(numberLayer), styleGrammar.get(numberLayer), styleVal));
            numberLayer--;
        }

        layers.get(layers.size() - 1).generateError(new NNTensor[]{errorContent});
        for (int i = layers.size() - 2; i >= 0; i--) {
            if (numberLayer >= 0 && layers.get(i) == styleLayers.get(numberLayer)) {
                layers.get(i + 1).getError()[0].add(errorGrammar(styleLayers.get(numberLayer), styleGrammar.get(numberLayer), styleVal));
                numberLayer--;
            }
            layers.get(i).generateError(layers.get(i + 1).getError());
        }
        delta.add(network.getError()[0]);
        optimizer.update();

        return accuracy;
    }

    private NNMatrix errorGrammar(NeuralLayer layer, NNMatrix grammar, double styleVal) {
        float styleV = (float) (styleVal / styleLayers.size());
        int[] size = layer.size();
        NNMatrix style_result = new NNMatrix(size[1] * size[0], size[2], layer.getOutput()[0].getData());
        NNMatrix grammar_result = grammarMatrix(style_result);

        accuracy += loss.findAccuracy(new NNMatrix[]{grammar_result}, new NNMatrix[]{grammar}) * styleV / style_result.size();
        NNMatrix errorGrammar = NNArrays.toMatrix(
                loss.findDerivative(new NNMatrix[]{grammar_result}, new NNMatrix[]{grammar}),
                grammar_result.shape()[0], grammar_result.shape()[1])[0];
        errorGrammar.mul(styleV / style_result.size());
        return derGrammarMatrix(style_result, errorGrammar);
    }
}
