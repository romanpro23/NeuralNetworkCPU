package neural_network.network.nlp;

import neural_network.activation.FunctionActivation;
import neural_network.layers.NeuralLayer;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.layer_2d.*;
import neural_network.layers.reshape.EmbeddingLayer;
import neural_network.layers.reshape.FlattenLayer2D;
import neural_network.layers.reshape.ImagePatchesLayer;
import neural_network.network.NeuralNetwork;

import static utilities.Use.WordCount;

public class TransformerVisual {
    private final NeuralNetwork transformer;
    private int depth;

    public TransformerVisual() {
        transformer = new NeuralNetwork();
    }

    public TransformerVisual addInputLayer(int row, int width, int depth) {
        transformer.addInputLayer(row, width, depth);
        this.depth = depth;

        return this;
    }

    public TransformerVisual addInputLayer(int width, int depth) {
        transformer.addInputLayer(width, depth);
        this.depth = depth;

        return this;
    }

    public TransformerVisual addLayer(NeuralLayer layer) {
        transformer.addLayer(layer);

        return this;
    }

    public TransformerVisual addEmbeddingLayer(int sizeVocabulary, int lengthVector) {
        transformer.addLayer(new EmbeddingLayer(sizeVocabulary, lengthVector));

        return this;
    }

    public TransformerVisual addImagePatchesLayer(int sizeKernel, int lengthVector) {
        transformer.addLayer(new ImagePatchesLayer(sizeKernel, lengthVector));

        return this;
    }

    public TransformerVisual addTYPE2Float3DLayer() {
        transformer.addLayer(new TYPE2Float3D());

        return this;
    }

    public TransformerVisual addTYPE2Float2DLayer() {
        transformer.addLayer(new TYPE2Float2D());

        return this;
    }


    public TransformerVisual addPositionalEmbedding() {
        transformer.addLayer(new PositionalEmbeddingLayer());

        return this;
    }

    public TransformerVisual addVITPositionalEmbedding() {
        transformer.addLayer(new VITPositionalEmbeddingLayer(false));

        return this;
    }

    public TransformerVisual addEncoderBlock(int countHead) {
        return addEncoderBlock(countHead, depth, depth, (float)0, (float)0);
    }

    public TransformerVisual addEncoderBlock(int countHead, int sizeDense) {
        return addEncoderBlock(countHead, depth, sizeDense, (float)0.0f, (float)0);
    }

    public TransformerVisual addEncoderBlock(int countHead, int sizeKey, int sizeDense) {
        return addEncoderBlock(countHead, sizeKey, sizeDense, (float)0, (float)0);
    }

    public TransformerVisual addEncoderBlock(int countHead, int sizeKey, int sizeDense, short dropout) {
        return addEncoderBlock(countHead, sizeKey, sizeDense, (float)0, dropout);
    }

    public TransformerVisual addEncoderBlock(int countHead, int sizeKey, int sizeDense, float dropoutAttention, float dropout) {
        transformer.addLayer(new AdditionBlock()
            .addLayer(new MultiHeadAttentionLayer(countHead, sizeKey, false, dropoutAttention))
        ).addLayer(new NormalizationLayer2D(false));

        AdditionBlock additionBlock = new AdditionBlock()
            .addLayer(new DenseLayer2D(sizeDense * 2, false))
            .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(), false))
            .addLayer(new DenseLayer2D(sizeDense, false));

        if (dropout != 0) {
            additionBlock.addLayer(new DropoutLayer2D(dropout));
        }

        transformer.addLayer(additionBlock)
            .addLayer(new NormalizationLayer2D(false));

        return this;
    }

    public TransformerVisual addDecoderBlock(int countHead) {
        return addDecoderBlock(countHead, depth, depth, (float)0, ((float)0));
    }

    public TransformerVisual addDecoderBlock(int countHead, int sizeDense) {
        return addDecoderBlock(countHead, depth, sizeDense, (float)0, ((float)0));
    }

    public TransformerVisual addDecoderBlock(int countHead, int sizeKey, int sizeDense) {
        return addDecoderBlock(countHead, sizeKey, sizeDense, (float)0, ((float)0));
    }

    public TransformerVisual addDecoderBlock(int countHead, int sizeKey, int sizeDense, float dropout) {
        return addDecoderBlock(countHead, sizeKey, sizeDense, (float)0, dropout);
    }

    public TransformerVisual addDecoderBlock(int countHead, int sizeKey, int sizeDense, float dropoutAttention, float dropout) {
        transformer.addLayer(new AdditionBlock()
            .addLayer(new MultiHeadAttentionLayer(countHead, sizeKey, false, dropoutAttention).setMask())
        ).addLayer(new NormalizationLayer2D(false))
        .addLayer(new AdditionBlock()
            .addLayer(new MultiHeadAttentionLayer(countHead, sizeKey, false, dropoutAttention))
        ).addLayer(new NormalizationLayer2D(false));

        AdditionBlock additionBlock = new AdditionBlock()
            .addLayer(new DenseLayer2D(sizeDense * 2, false))
            .addLayer(new ActivationLayer2D(new FunctionActivation.GELU(), false))
            .addLayer(new DenseLayer2D(sizeDense, false));

        if (dropout != 0) {
            additionBlock.addLayer(new DropoutLayer2D(dropout));
        }

        transformer.addLayer(additionBlock)
            .addLayer(new NormalizationLayer2D(false));

        return this;
    }

    public TransformerVisual addDenseLayer(int countNeuron){
        transformer.addLayer(new DenseLayer2D(countNeuron, false));

        return this;
    }

    public TransformerVisual addFlattenLayer(){
        transformer.addLayer(new FlattenLayer2D(false));

        return this;
    }

    public TransformerVisual addDenseLayer1D(int countNeuron){
        transformer.addLayer(new DenseLayer(countNeuron, false));

        return this;
    }

    public TransformerVisual addSoftmax(){
        transformer.addLayer(new SoftmaxLayer2D());

        return this;
    }

    public NeuralNetwork createTransformer(){
        return transformer;
    }
}
