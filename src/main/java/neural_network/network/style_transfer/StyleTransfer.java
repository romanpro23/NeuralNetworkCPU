package neural_network.network.style_transfer;

import data.ImageCreator;
import lombok.NoArgsConstructor;
import neural_network.layers.NeuralLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
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

    private ArrayList<NeuralLayer> contentLayers;
    private ArrayList<NNTensor> contentImg;

    private NNTensor result;
    private NNTensor delta;
    private float accuracy;

    private boolean variation;
    private float variationLoss;

    private NNTensor verticalSobel;
    private NNTensor horizontalSobel;
    private NNTensor content;

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
        contentLayers = new ArrayList<>();
        styleGrammar = new ArrayList<>();
        contentImg = new ArrayList<>();

        this.loss = new FunctionLoss.MSE();
    }

    public StyleTransfer addContentLayer(NeuralLayer... neuralLayers) {
        Collections.addAll(this.contentLayers, neuralLayers);

        return this;
    }

    public StyleTransfer addStyleLayer(NeuralLayer... neuralLayers) {
        Collections.addAll(this.styleLayers, neuralLayers);

        return this;
    }

    public StyleTransfer setContent(NNTensor content) {
        if (contentLayers.isEmpty()) {
            contentLayers.add(network.getLastLayer());
        }

        network.query(new NNTensor[]{content});
        for (NeuralLayer contentLayer : contentLayers) {
            contentImg.add(NNArrays.isTensor(contentLayer.getOutput())[0]);
        }

        result = new NNTensor(content.shape());
        result.add(content);
        delta = new NNTensor(content.shape());

        return this;
    }

    public StyleTransfer setResult(NNTensor result) {
        this.result = result;

        return this;
    }

    public StyleTransfer addVariationLoss(double variationLoss) {
        this.variationLoss = (float) variationLoss;
        this.variation = true;

        return this;
    }

    public StyleTransfer setStyle(NNTensor style) {
        network.query(new NNTensor[]{style});

        for (NeuralLayer styleLayer : styleLayers) {
            int[] size = styleLayer.size();
            NNMatrix style_result = new NNMatrix(size[1] * size[0], size[2], styleLayer.getOutput()[0].getData(), styleLayer.getOutput()[0].getSdata());

            styleGrammar.add(grammarMatrix(style_result));
        }

        return this;
    }

    public StyleTransfer setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;

        return this;
    }

    public StyleTransfer create() {
        optimizer.addDataOptimize(result, delta);
        if(variation){
            horizontalSobel = ImageCreator.horizontalSobelEdge(content);
        }
        return this;
    }

    public NNTensor getResult() {
        return result;
    }

    public float train(double styleVal, double contentVal) {
        accuracy = 0;
        network.query(new NNTensor[]{result});

        ArrayList<NeuralLayer> layers = network.getLayers();
        int numberLayerStyle = styleLayers.size() - 1;
        int numberLayerContent = contentLayers.size() - 1;

        NNTensor error = new NNTensor(network.getOutputSize());
        if (layers.get(layers.size() - 1) == styleLayers.get(numberLayerStyle)) {
            error.add(errorStyle(styleLayers.get(numberLayerStyle),
                    styleGrammar.get(numberLayerStyle),
                    styleVal));
            numberLayerStyle--;
        }
        if (layers.get(layers.size() - 1) == contentLayers.get(numberLayerContent)) {
            error.add(errorContent(contentLayers.get(numberLayerContent),
                    contentImg.get(numberLayerContent),
                    contentVal));
            numberLayerContent--;
        }
        layers.get(layers.size() - 1).generateError(new NNTensor[]{error});

        for (int i = layers.size() - 2; i >= 0; i--) {
            if (numberLayerStyle >= 0 && layers.get(i) == styleLayers.get(numberLayerStyle)) {
                layers.get(i + 1).getError()[0].add(errorStyle(styleLayers.get(numberLayerStyle),
                        styleGrammar.get(numberLayerStyle),
                        styleVal));
                numberLayerStyle--;
            }
            if (numberLayerContent >= 0 && layers.get(i) == contentLayers.get(numberLayerContent)) {
                layers.get(i + 1).getError()[0].add(errorContent(contentLayers.get(numberLayerContent),
                        contentImg.get(numberLayerContent),
                        contentVal));
                numberLayerContent--;
            }
            layers.get(i).generateError(layers.get(i + 1).getError());
        }
        delta.add(network.getError()[0]);
        optimizer.update();

        return accuracy;
    }

    private NNMatrix errorStyle(NeuralLayer layer, NNMatrix grammar, double styleVal) {
        float styleV = (float) (styleVal / styleLayers.size());
        int[] size = layer.size();
        NNMatrix style_result = new NNMatrix(size[1] * size[0], size[2], layer.getOutput()[0].getData(), layer.getOutput()[0].getSdata());
        NNMatrix grammar_result = grammarMatrix(style_result);

        accuracy += loss.findAccuracy(new NNMatrix[]{grammar_result}, new NNMatrix[]{grammar}) * styleV / style_result.size();
        NNMatrix errorGrammar = NNArrays.toMatrix(
                loss.findDerivative(new NNMatrix[]{grammar_result}, new NNMatrix[]{grammar}),
                grammar_result.shape()[0], grammar_result.shape()[1])[0];
        errorGrammar.mul(styleV / style_result.size());
        return derGrammarMatrix(style_result, errorGrammar);
    }

    private NNArray errorContent(NeuralLayer layer, NNTensor content, double contentVal) {
        float val = (float) (contentVal / contentLayers.size());
        accuracy += loss.findAccuracy(layer.getOutput(), new NNTensor[]{content}) * val;
        return loss.findDerivative(layer.getOutput(), new NNTensor[]{content})[0].mul(val);
    }
}
