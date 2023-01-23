package neural_network.network.classification;

import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.NeuralLayer;
import neural_network.layers.layer_3d.ActivationLayer3D;
import neural_network.layers.layer_3d.BatchNormalizationLayer3D;
import neural_network.layers.layer_3d.ConvolutionLayer;
import neural_network.layers.layer_3d.MaxPoolingLayer;
import neural_network.layers.layer_1d.ActivationLayer;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.layer_1d.DropoutLayer;
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
        vgg.addLayer(new MaxPoolingLayer(2, 2, 0));

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

    public NeuralNetwork createVGG16(int height, int width, int depth, double scaleWidth, int sizeHidden, int sizeOutput, double dropout) {
        return createVGG16(height, width, depth, scaleWidth, sizeHidden, sizeOutput, dropout, false);
    }

    public NeuralNetwork createVGG16(int height, int width, int depth, double scaleWidth, int sizeHidden, int sizeOutput, double dropout, boolean batchnorm) {
        addInputLayer(height, width, depth);
        addConvolutionLayer((int) (64 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (64 * scaleWidth), 3, batchnorm);
        addMaxPoolingLayer();
        addConvolutionLayer((int) (128 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (128 * scaleWidth), 3, batchnorm);
        addMaxPoolingLayer();
        addConvolutionLayer((int) (256 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (256 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (256 * scaleWidth), 3, batchnorm);
        addMaxPoolingLayer();
        addConvolutionLayer((int) (512 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (512 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (512 * scaleWidth), 3, batchnorm);
        addMaxPoolingLayer();
        addConvolutionLayer(512, 3, batchnorm);
        addConvolutionLayer(512, 3, batchnorm);
        addConvolutionLayer(512, 3, batchnorm);
        addMaxPoolingLayer();
        addDenseLayer(sizeHidden, new FunctionActivation.ReLU());
        addDropoutLayer(dropout);
        addDenseLayer(sizeHidden, new FunctionActivation.ReLU());
        addDropoutLayer(dropout);
        addDenseLayer(sizeOutput, new FunctionActivation.Softmax());

        return createVGG();
    }

    public NeuralNetwork createVGG19(int height, int width, int depth, double scaleWidth, int sizeHidden, int sizeOutput, double dropout) {
        return createVGG19(height, width, depth, scaleWidth, sizeHidden, sizeOutput, dropout, false);
    }

    public NeuralNetwork createVGG19(int height, int width, int depth, double scaleWidth, int sizeHidden, int sizeOutput, double dropout, boolean batchnorm) {
        addInputLayer(height, width, depth);
        addConvolutionLayer((int) (64 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (64 * scaleWidth), 3, batchnorm);
        addMaxPoolingLayer();
        addConvolutionLayer((int) (128 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (128 * scaleWidth), 3, batchnorm);
        addMaxPoolingLayer();
        addConvolutionLayer((int) (256 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (256 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (256 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (256 * scaleWidth), 3, batchnorm);
        addMaxPoolingLayer();
        addConvolutionLayer((int) (512 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (512 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (512 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (512 * scaleWidth), 3, batchnorm);
        addMaxPoolingLayer();
        addConvolutionLayer(512, 3, batchnorm);
        addConvolutionLayer(512, 3, batchnorm);
        addConvolutionLayer(512, 3, batchnorm);
        addConvolutionLayer(512, 3, batchnorm);
        addMaxPoolingLayer();
        addDenseLayer(sizeHidden, new FunctionActivation.ReLU());
        addDropoutLayer(dropout);
        addDenseLayer(sizeHidden, new FunctionActivation.ReLU());
        addDropoutLayer(dropout);
        addDenseLayer(sizeOutput, new FunctionActivation.Softmax());

        return createVGG();
    }

    public NeuralNetwork createVGG13(int height, int width, int depth, double scaleWidth, int sizeHidden, int sizeOutput, double dropout) {
        return createVGG13(height, width, depth, scaleWidth, sizeHidden, sizeOutput, dropout, false);
    }

    public NeuralNetwork createVGG13(int height, int width, int depth, double scaleWidth, int sizeHidden, int sizeOutput, double dropout, boolean batchnorm) {
        addInputLayer(height, width, depth);
        addConvolutionLayer((int) (64 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (64 * scaleWidth), 3, batchnorm);
        addMaxPoolingLayer();
        addConvolutionLayer((int) (128 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (128 * scaleWidth), 3, batchnorm);
        addMaxPoolingLayer();
        addConvolutionLayer((int) (256 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (256 * scaleWidth), 3, batchnorm);
        addMaxPoolingLayer();
        addConvolutionLayer((int) (512 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (512 * scaleWidth), 3, batchnorm);
        addMaxPoolingLayer();
        addConvolutionLayer(512, 3, batchnorm);
        addConvolutionLayer(512, 3, batchnorm);
        addMaxPoolingLayer();
        addDenseLayer(sizeHidden, new FunctionActivation.ReLU());
        addDropoutLayer(dropout);
        addDenseLayer(sizeHidden, new FunctionActivation.ReLU());
        addDropoutLayer(dropout);
        addDenseLayer(sizeOutput, new FunctionActivation.Softmax());

        return createVGG();
    }

    public NeuralNetwork createVGG11(int height, int width, int depth, double scaleWidth, int sizeHidden, int sizeOutput, double dropout) {
        return createVGG11(height, width, depth, scaleWidth, sizeHidden, sizeOutput, dropout, false);
    }

    public NeuralNetwork createVGG11(int height, int width, int depth, double scaleWidth, int sizeHidden, int sizeOutput, double dropout, boolean batchnorm) {
        addInputLayer(height, width, depth);
        addConvolutionLayer((int) (64 * scaleWidth), 3, batchnorm);
        addMaxPoolingLayer();
        addConvolutionLayer((int) (128 * scaleWidth), 3, batchnorm);
        addMaxPoolingLayer();
        addConvolutionLayer((int) (256 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (256 * scaleWidth), 3, batchnorm);
        addMaxPoolingLayer();
        addConvolutionLayer((int) (512 * scaleWidth), 3, batchnorm);
        addConvolutionLayer((int) (512 * scaleWidth), 3, batchnorm);
        addMaxPoolingLayer();
        addConvolutionLayer(512, 3, batchnorm);
        addConvolutionLayer(512, 3, batchnorm);
        addMaxPoolingLayer();
        addDenseLayer(sizeHidden, new FunctionActivation.ReLU());
        addDropoutLayer(dropout);
        addDenseLayer(sizeHidden, new FunctionActivation.ReLU());
        addDropoutLayer(dropout);
        addDenseLayer(sizeOutput, new FunctionActivation.Softmax());

        return createVGG();
    }
}
