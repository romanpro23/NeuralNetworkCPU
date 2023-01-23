package neural_network.network.classification;

import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.NeuralLayer;
import neural_network.layers.layer_3d.*;
import neural_network.layers.layer_3d.inception.InceptionBlock;
import neural_network.layers.layer_3d.inception.InceptionUnit;
import neural_network.layers.layer_1d.DropoutLayer;
import neural_network.layers.reshape.GlobalAveragePooling3DLayer;
import neural_network.network.NeuralNetwork;

public class Inception {
    private final NeuralNetwork inception;

    public Inception() {
        this.inception = new NeuralNetwork();
    }

    public Inception addInputLayer(int height, int width, int depth) {
        inception.addInputLayer(height, width, depth);

        return this;
    }

    public Inception addLayer(NeuralLayer layer) {
        inception.addLayer(layer);

        return this;
    }

    public Inception addConvolutionLayer(int countKernel, int sizeKernel) {
        return addConvolutionLayer(countKernel, sizeKernel, 1, sizeKernel / 2);
    }

    public Inception addConvolutionLayer(int countKernel, int sizeKernel, int stride, int padding) {
        inception.addLayer(new ConvolutionLayer(countKernel, sizeKernel, stride, padding).setInitializer(new Initializer.HeNormal()));
        inception.addLayer(new BatchNormalizationLayer3D(0.9));
        inception.addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()));

        return this;
    }

    public Inception addMaxPoolingLayer(int sizeKernel, int stride) {
        inception.addLayer(new MaxPoolingLayer(sizeKernel, stride, sizeKernel / 2));

        return this;
    }

    public Inception addInceptionA(int poolFeatures) {
        return addInceptionA(poolFeatures, 1);
    }

    public Inception addInceptionA(int poolFeatures, int scale) {
        inception.addLayer(
                new InceptionBlock()
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new ConvolutionLayer(64 / scale, 1, 1, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                        )
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new ConvolutionLayer(48 / scale, 1, 1, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(64 / scale, 5, 1, 2).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                        )
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new ConvolutionLayer(64 / scale, 1, 1, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(96 / scale, 3, 1, 1).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(96 / scale, 3, 1, 1).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                        )
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new AveragePoolingLayer(3, 1, 1))
                                .addLayer(new ConvolutionLayer(poolFeatures / scale, 1, 1, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                        )
        );

        return this;
    }

    public Inception addInceptionB() {
        return addInceptionB(1);
    }

    public Inception addInceptionB(int scale) {
        inception.addLayer(
                new InceptionBlock()
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new ConvolutionLayer(384 / scale, 3, 2, 1).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                        )
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new ConvolutionLayer(64 / scale, 1, 1, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(96 / scale, 3, 1, 1).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(96 / scale, 3, 2, 1).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                        )
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new MaxPoolingLayer(3, 2, 1))
                        )
        );

        return this;
    }

    public Inception addInceptionC(int channels) {
        return addInceptionC(7, channels);
    }

    public Inception addInceptionC(int sizeKernel, int channels) {
        return addInceptionC(sizeKernel, channels, 1);
    }

    public Inception addInceptionC(int sizeKernel, int channels, int scale) {
        inception.addLayer(
                new InceptionBlock()
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new ConvolutionLayer(192 / scale, 1, 1, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                        )
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new ConvolutionLayer(channels / scale, 1, 1, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(channels / scale, 1, sizeKernel, 1, 0, sizeKernel / 2).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(192 / scale, sizeKernel, 1, 1, sizeKernel / 2, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                        )
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new ConvolutionLayer(channels / scale, 1, 1, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(channels / scale, 1, sizeKernel, 1, 0, sizeKernel / 2).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(channels / scale, sizeKernel, 1, 1, sizeKernel / 2, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(channels / scale, 1, sizeKernel, 1, 0, sizeKernel / 2).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(192 / scale, sizeKernel, 1, 1, sizeKernel / 2, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                        )
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new AveragePoolingLayer(3, 1, 1))
                                .addLayer(new ConvolutionLayer(192 / scale, 1, 1, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                        )
        );

        return this;
    }

    public Inception addInceptionD(int sizeKernel) {
        return addInceptionD(sizeKernel, 1);
    }

    public Inception addInceptionD(int sizeKernel, int scale) {
        inception.addLayer(
                new InceptionBlock()
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new ConvolutionLayer(192 / scale, 1, 1, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(320 / scale, 3, 2, 1).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                        )
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new ConvolutionLayer(192 / scale, 1, 1, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(192 / scale, 1, sizeKernel, 1, 0, sizeKernel / 2).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(192 / scale, sizeKernel, 1, 1, sizeKernel / 2, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(192 / scale, 3, 2, 1).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                        )
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new MaxPoolingLayer(3, 2, 1))
                        )
        );

        return this;
    }

    public Inception addInceptionE() {
        return addInceptionE(1);
    }

    public Inception addInceptionE(int scale) {
        inception.addLayer(
                new InceptionBlock()
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new ConvolutionLayer(320 / scale, 1, 1, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                        )
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new ConvolutionLayer(384 / scale, 1, 1, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new InceptionBlock()
                                        .addInceptionUnit(new InceptionUnit()
                                                .addLayer(new ConvolutionLayer(384 / scale, 1, 3, 1, 0, 1).setInitializer(new Initializer.HeNormal()))
                                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                        )
                                        .addInceptionUnit(new InceptionUnit()
                                                .addLayer(new ConvolutionLayer(384 / scale, 3, 1, 1, 1, 0).setInitializer(new Initializer.HeNormal()))
                                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                        )
                                )
                        )
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new ConvolutionLayer(448 / scale, 1, 1, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(384 / scale, 3, 1, 1).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new InceptionBlock()
                                        .addInceptionUnit(new InceptionUnit()
                                                .addLayer(new ConvolutionLayer(384 / scale, 1, 3, 1, 0, 1).setInitializer(new Initializer.HeNormal()))
                                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                        )
                                        .addInceptionUnit(new InceptionUnit()
                                                .addLayer(new ConvolutionLayer(384 / scale, 3, 1, 1, 1, 0).setInitializer(new Initializer.HeNormal()))
                                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                        )
                                )
                        )
                        .addInceptionUnit(new InceptionUnit()
                                .addLayer(new AveragePoolingLayer(3, 1, 1))
                                .addLayer(new ConvolutionLayer(192 / scale, 1, 1, 0).setInitializer(new Initializer.HeNormal()))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                        )
        );

        return this;
    }

    public Inception addDenseLayer(int countNeuron, FunctionActivation functionActivation) {
        inception.addDenseLayer(countNeuron, functionActivation);

        return this;
    }

    public Inception addGlobalAveragePoolingLayer() {
        inception.addLayer(new GlobalAveragePooling3DLayer());

        return this;
    }

    public Inception addDropoutLayer(double dropout) {
        inception.addLayer(new DropoutLayer(dropout));

        return this;
    }

    public NeuralNetwork createInception() {
        return inception;
    }
}
