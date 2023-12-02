package neural_network.network.classification;

import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.NeuralLayer;
import neural_network.layers.layer_3d.*;
import neural_network.layers.layer_3d.densely.DenseBlock;
import neural_network.layers.layer_3d.densely.DenseUnit;
import neural_network.layers.layer_1d.ActivationLayer;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.reshape.FlattenLayer3D;
import neural_network.layers.reshape.GlobalAveragePoolingLayer3D;
import neural_network.network.NeuralNetwork;

public class DenseNet {
    private final NeuralNetwork densenet;

    public DenseNet() {
        this.densenet = new NeuralNetwork();
    }

    public DenseNet addInputLayer(int height, int width, int depth) {
        densenet.addInputLayer(height, width, depth);

        return this;
    }

    public DenseNet addLayer(NeuralLayer layer) {
        densenet.addLayer(layer);

        return this;
    }

    public DenseNet addConvolutionLayer(int countKernel, int sizeKernel) {
        return addConvolutionLayer(countKernel, sizeKernel, 1, sizeKernel / 2);
    }

    public DenseNet addConvolutionLayer(int countKernel, int sizeKernel, int stride, int padding) {
        densenet.addLayer(new ConvolutionLayer(countKernel, sizeKernel, stride, padding).setInitializer(new Initializer.HeNormal()));
        densenet.addLayer(new BatchNormalizationLayer3D(0.9));
        densenet.addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()));

        return this;
    }

    public DenseNet addMaxPoolingLayer(int sizeKernel, int stride) {
        densenet.addLayer(new MaxPoolingLayer(sizeKernel, stride, sizeKernel / 2));

        return this;
    }

    public DenseNet addGlobalAveragePoolingLayer() {
        densenet.addLayer(new BatchNormalizationLayer3D())
                .addLayer(new GlobalAveragePoolingLayer3D());
        return this;
    }

    public DenseNet addDenseLayer(int countNeuron, FunctionActivation functionActivation) {
        if (densenet.getOutputSize().length != 1) {
            densenet.addLayer(new FlattenLayer3D());
        }
        densenet.addLayer(new DenseLayer(countNeuron, false).setInitializer(new Initializer.XavierNormal()));
        densenet.addLayer(new ActivationLayer(functionActivation));

        return this;
    }

    private DenseUnit getDenseUnit(int countKernel, int bottleNeck, double dropout) {
        DenseUnit denseUnit = new DenseUnit()
                .addLayer(new BatchNormalizationLayer3D())
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new ConvolutionLayer(countKernel * bottleNeck, 1, 1))
                .addLayer(new BatchNormalizationLayer3D())
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new ConvolutionLayer(countKernel, 3, 1, 1));

        if (dropout > 0) {
            denseUnit.addLayer(new DropoutLayer3D(dropout));
        }

        return denseUnit;
    }

    public DenseNet addDenseBlock(int countLayers, int countKernel) {
        return addDenseBlock(countLayers, countKernel, 4, 0);
    }

    public DenseNet addDenseBlock(int countLayers, int countKernel, int bottleNeck) {
        return addDenseBlock(countLayers, countKernel, bottleNeck, 0);
    }

    public DenseNet addDenseBlock(int countLayers, int countKernel, int bottleNeck, double dropout) {
        DenseBlock block = new DenseBlock();
        for (int i = 0; i < countLayers; i++) {
            block.addDenseUnit(getDenseUnit(countKernel, bottleNeck, dropout));
        }
        densenet.addLayer(block);

        return this;
    }

    public DenseNet addTransition(int countKernel) {
        densenet.addLayer(new BatchNormalizationLayer3D())
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new ConvolutionLayer(countKernel, 1, 1, 0))
                .addLayer(new AveragePoolingLayer(2, 2));

        return this;
    }

    public NeuralNetwork createDenseNet() {
        return densenet;
    }
}