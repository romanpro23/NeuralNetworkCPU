package neural_network.network.classification;

import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.NeuralLayer;
import neural_network.layers.convolution_3d.ActivationLayer3D;
import neural_network.layers.convolution_3d.BatchNormalizationLayer3D;
import neural_network.layers.convolution_3d.ConvolutionLayer;
import neural_network.layers.convolution_3d.MaxPoolingLayer;
import neural_network.layers.dense.ActivationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.layers.dense.DropoutLayer;
import neural_network.layers.reshape.Flatten3DLayer;
import neural_network.network.NeuralNetwork;

public class VGG {
    private final NeuralNetwork vgg;

    public VGG() {
        this.vgg = new NeuralNetwork();
    }

    public VGG addInputLayer(int height, int width, int depth) {
        vgg.addInputLayer(height, width, depth);

        return this;
    }

    public VGG addLayer(NeuralLayer layer) {
        vgg.addLayer(layer);

        return this;
    }

    public VGG addConvolutionLayer(int countKernel, int sizeKernel) {
        return addConvolutionLayer(countKernel, sizeKernel, false);
    }

    public VGG addConvolutionLayer(int countKernel, int sizeKernel, boolean batchnorm) {
        vgg.addLayer(new ConvolutionLayer(countKernel, sizeKernel, 1, sizeKernel / 2)
                .setInitializer(new Initializer.HeNormal()));
        if (batchnorm) {
            vgg.addLayer(new BatchNormalizationLayer3D(0.9));
        }
        vgg.addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()));

        return this;
    }

    public VGG addMaxPoolingLayer() {
        vgg.addLayer(new MaxPoolingLayer(2));

        return this;
    }

    public VGG addDenseLayer(int countNeuron, FunctionActivation functionActivation) {
        if (vgg.getOutputSize().length != 1) {
            vgg.addLayer(new Flatten3DLayer());
        }
        vgg.addLayer(new DenseLayer(countNeuron)
                .setInitializer(new Initializer.XavierNormal()));
        vgg.addLayer(new ActivationLayer(functionActivation));

        return this;
    }

    public VGG addDropoutLayer(double dropout) {
        vgg.addLayer(new DropoutLayer(dropout));

        return this;
    }

    public NeuralNetwork createVGG() {
        return vgg;
    }
}
