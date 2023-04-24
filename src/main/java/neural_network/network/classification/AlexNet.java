package neural_network.network.classification;

import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.NeuralLayer;
import neural_network.layers.layer_3d.*;
import neural_network.layers.layer_1d.ActivationLayer;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.layer_1d.DropoutLayer;
import neural_network.layers.reshape.FlattenLayer3D;
import neural_network.network.NeuralNetwork;
import neural_network.regularization.Regularization;

public class AlexNet {
    private final NeuralNetwork alexnet;

    public AlexNet() {
        this.alexnet = new NeuralNetwork();
    }

    public AlexNet addInputLayer(int height, int width, int depth) {
        alexnet.addInputLayer(height, width, depth);

        return this;
    }

    public AlexNet addLayer(NeuralLayer layer) {
        alexnet.addLayer(layer);

        return this;
    }

    public AlexNet addConvolutionLayer(int countKernel, int sizeKernel, int step, int padding, int countGroup) {
        alexnet.addLayer(new GroupedConvolutionLayer(countKernel, sizeKernel, step, padding, countGroup)
                .setInitializer(new Initializer.HeNormal())
                .setRegularization(new Regularization.L2(0.0005)))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()));

        return this;
    }

    public AlexNet addConvolutionLayer(int countKernel, int sizeKernel, int step, int padding) {
        alexnet.addLayer(new ConvolutionLayer(countKernel, sizeKernel, step, padding)
                .setInitializer(new Initializer.HeNormal())
                .setRegularization(new Regularization.L2(0.0005)))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()));

        return this;
    }

    public AlexNet addConvolutionLayer(int countKernel, int sizeKernel, int step, int padding, boolean batchnorm) {
        alexnet.addLayer(new ConvolutionLayer(countKernel, sizeKernel, step, padding)
                .setInitializer(new Initializer.HeNormal())
                .setRegularization(new Regularization.L2(0.0005)));
        if (batchnorm) {
            alexnet.addLayer(new BatchNormalizationLayer3D(0.9));
        }
        alexnet.addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()));

        return this;
    }

    public AlexNet addMaxPoolingLayer(int kernelSize, int stride) {
        alexnet.addLayer(new MaxPoolingLayer(kernelSize, stride, 0));

        return this;
    }

    public AlexNet addMaxPoolingLayer(int kernelSize, int stride, int padding) {
        alexnet.addLayer(new MaxPoolingLayer(kernelSize, stride, padding));

        return this;
    }

    public AlexNet addDenseLayer(int countNeuron, FunctionActivation functionActivation) {
        if (alexnet.getOutputSize().length != 1) {
            alexnet.addLayer(new FlattenLayer3D());
        }
        alexnet.addLayer(new DenseLayer(countNeuron));
        alexnet.addLayer(new ActivationLayer(functionActivation));

        return this;
    }

    public AlexNet addDropoutLayer(double dropout) {
        alexnet.addLayer(new DropoutLayer(dropout));

        return this;
    }

    public NeuralNetwork createAlexNet() {
        return alexnet;
    }

    public NeuralNetwork createAlexNet(int height, int width, int depth, double scaleWidth, int countGroup, int sizeHidden, int sizeOutput, double dropout) {
        addInputLayer(height, width, depth);
        addConvolutionLayer((int) (96 * scaleWidth), 11, 4, 0);
        addMaxPoolingLayer(3, 2);
        addConvolutionLayer((int) (256 * scaleWidth), 5, 1, 2, countGroup);
        addMaxPoolingLayer(3, 2);
        addConvolutionLayer((int) (384 * scaleWidth), 3, 1, 1, countGroup);
        addConvolutionLayer((int) (384 * scaleWidth), 3, 1, 1, countGroup);
        addConvolutionLayer((int) (256 * scaleWidth), 3, 1, 1, countGroup);
        addMaxPoolingLayer(3, 2);
        addLayer(new DropoutLayer3D(dropout));
        addDenseLayer(sizeHidden, new FunctionActivation.ReLU());
        addDropoutLayer(dropout);
        addDenseLayer(sizeHidden, new FunctionActivation.ReLU());
        addDenseLayer(sizeOutput, new FunctionActivation.Softmax());

        return createAlexNet();
    }

    public NeuralNetwork createAlexNet(int height, int width, int depth, double scaleWidth, int sizeHidden, int sizeOutput, double dropout) {
        addInputLayer(height, width, depth);
        addConvolutionLayer((int) (96 * scaleWidth), 11, 4, 0);
        addMaxPoolingLayer(3, 2);
        addConvolutionLayer((int) (256 * scaleWidth), 5, 1, 2);
        addMaxPoolingLayer(3, 2);
        addConvolutionLayer((int) (384 * scaleWidth), 3, 1, 1);
        addConvolutionLayer((int) (384 * scaleWidth), 3, 1, 1);
        addConvolutionLayer((int) (256 * scaleWidth), 3, 1, 1);
        addMaxPoolingLayer(3, 2);
        addLayer(new DropoutLayer3D(dropout));
        addDenseLayer(sizeHidden, new FunctionActivation.ReLU());
        addDropoutLayer(dropout);
        addDenseLayer(sizeHidden, new FunctionActivation.ReLU());
        addDenseLayer(sizeOutput, new FunctionActivation.Softmax());

        return createAlexNet();
    }
}
