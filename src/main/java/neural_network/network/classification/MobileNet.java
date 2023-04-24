package neural_network.network.classification;

import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.NeuralLayer;
import neural_network.layers.layer_3d.*;
import neural_network.layers.layer_3d.residual.ResidualBlock;
import neural_network.layers.layer_3d.residual.ResidualUnit;
import neural_network.layers.layer_1d.ActivationLayer;
import neural_network.layers.layer_1d.DenseLayer;
import neural_network.layers.layer_1d.DropoutLayer;
import neural_network.layers.reshape.FlattenLayer3D;
import neural_network.layers.reshape.GlobalAveragePoolingLayer3D;
import neural_network.network.NeuralNetwork;

public class MobileNet {
    private final NeuralNetwork mobilenet;
    private int countFilter;

    public MobileNet() {
        this.mobilenet = new NeuralNetwork();
    }

    public MobileNet addInputLayer(int height, int width, int depth) {
        mobilenet.addInputLayer(height, width, depth);
        countFilter = depth;

        return this;
    }

    public MobileNet addLayer(NeuralLayer layer) {
        mobilenet.addLayer(layer);

        return this;
    }

    public MobileNet addConvolutionLayer(int countKernel, int sizeKernel, int step) {
        mobilenet.addLayer(new ConvolutionLayer(countKernel, sizeKernel, step, sizeKernel / 2).setInitializer(new Initializer.HeNormal()))
                .addLayer(new BatchNormalizationLayer3D())
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLUMax(6)));
        countFilter = countKernel;

        return this;
    }

    public MobileNet addDWConvolutionLayer(int countKernel, int sizeKernel, int step) {
        mobilenet.addLayer(new GroupedConvolutionLayer(countKernel, sizeKernel, step, sizeKernel / 2, countKernel).setInitializer(new Initializer.HeNormal()))
                .addLayer(new BatchNormalizationLayer3D())
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLUMax(6)));

        return this;
    }

    public MobileNet addDenseLayer(int countNeuron, FunctionActivation functionActivation) {
        if (mobilenet.getOutputSize().length != 1) {
            mobilenet.addLayer(new FlattenLayer3D());
        }
        mobilenet.addLayer(new DenseLayer(countNeuron)
                .setInitializer(new Initializer.XavierNormal()));
        mobilenet.addLayer(new ActivationLayer(functionActivation));

        return this;
    }

    public MobileNet addDropoutLayer(double dropout) {
        mobilenet.addLayer(new DropoutLayer(dropout));

        return this;
    }

    public MobileNet addGlobalAveragePoolingLayer() {
        mobilenet.addLayer(new GlobalAveragePoolingLayer3D());

        return this;
    }

    public MobileNet addBottleneck(int countKernel, int stride, int t) {
        if (stride == 1) {
            mobilenet.addLayer(
                    new ResidualBlock()
                            .addResidualUnit(new ResidualUnit())
                            .addResidualUnit(new ResidualUnit()
                                    .addLayer(new ConvolutionLayer(countFilter * t, 1, 1, 0)
                                            .setInitializer(new Initializer.HeNormal()))
                                    .addLayer(new BatchNormalizationLayer3D())
                                    .addLayer(new ActivationLayer3D(new FunctionActivation.ReLUMax(6)))
                                    .addLayer(new GroupedConvolutionLayer(countFilter * t, 3, stride, 1, countFilter * t)
                                            .setInitializer(new Initializer.HeNormal()))
                                    .addLayer(new BatchNormalizationLayer3D())
                                    .addLayer(new ActivationLayer3D(new FunctionActivation.ReLUMax(6)))
                                    .addLayer(new ConvolutionLayer(countKernel, 1, 1, 0)
                                            .setInitializer(new Initializer.HeNormal()))
                                    .addLayer(new BatchNormalizationLayer3D())
                            )
            );
        } else {
            mobilenet.addLayer(new ConvolutionLayer(countFilter * t, 1, 1, 0)
                    .setInitializer(new Initializer.HeNormal()))
                    .addLayer(new BatchNormalizationLayer3D())
                    .addLayer(new ActivationLayer3D(new FunctionActivation.ReLUMax(6)))
                    .addLayer(new GroupedConvolutionLayer(countFilter * t, 3, stride, 1, countFilter * t)
                            .setInitializer(new Initializer.HeNormal()))
                    .addLayer(new BatchNormalizationLayer3D())
                    .addLayer(new ActivationLayer3D(new FunctionActivation.ReLUMax(6)))
                    .addLayer(new ConvolutionLayer(countKernel, 1, 1, 0)
                            .setInitializer(new Initializer.HeNormal()))
                    .addLayer(new BatchNormalizationLayer3D());
        }

        countFilter = countKernel;
        return this;
    }

    public NeuralNetwork createMobilenet() {
        return mobilenet;
    }

    public NeuralNetwork createMobileNetV1(int height, int width, int depth, double scaleWidth, int sizeOutput, double dropout) {
        addInputLayer(height, width, depth)
                .addConvolutionLayer((int) (32 * scaleWidth), 3, 2)
                .addDWConvolutionLayer((int) (32 * scaleWidth), 3, 1)
                .addConvolutionLayer((int) (64 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (64 * scaleWidth), 3, 2)
                .addConvolutionLayer((int) (128 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (128 * scaleWidth), 3, 1)
                .addConvolutionLayer((int) (128 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (128 * scaleWidth), 3, 2)
                .addConvolutionLayer((int) (256 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (256 * scaleWidth), 3, 1)
                .addConvolutionLayer((int) (256 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (256 * scaleWidth), 3, 2)
                .addConvolutionLayer((int) (512 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (512 * scaleWidth), 3, 1)
                .addConvolutionLayer((int) (512 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (512 * scaleWidth), 3, 1)
                .addConvolutionLayer((int) (512 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (512 * scaleWidth), 3, 1)
                .addConvolutionLayer((int) (512 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (512 * scaleWidth), 3, 1)
                .addConvolutionLayer((int) (512 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (512 * scaleWidth), 3, 1)
                .addConvolutionLayer((int) (512 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (512 * scaleWidth), 3, 2)
                .addConvolutionLayer((int) (1024 * scaleWidth), 1, 1)
                .addDWConvolutionLayer((int) (1024 * scaleWidth), 3, 1)
                .addConvolutionLayer((int) (1024 * scaleWidth), 1, 1)
                .addGlobalAveragePoolingLayer()
                .addDropoutLayer(dropout)
                .addDenseLayer(sizeOutput, new FunctionActivation.Softmax());

        return createMobilenet();
    }

    public NeuralNetwork createMobileNetV2(int height, int width, int depth, double scaleWidth, int sizeOutput, double dropout) {
        addInputLayer(height, width, depth)
                .addConvolutionLayer((int) (32 * scaleWidth), 3, 2)
                .addBottleneck((int) (16 * scaleWidth), 1,1)
                .addBottleneck((int) (24 * scaleWidth), 2,6)
                .addBottleneck((int) (24 * scaleWidth), 1,6)
                .addBottleneck((int) (32 * scaleWidth), 2,6)
                .addBottleneck((int) (32 * scaleWidth), 1,6)
                .addBottleneck((int) (32 * scaleWidth), 1,6)
                .addBottleneck((int) (64 * scaleWidth), 2,6)
                .addBottleneck((int) (64 * scaleWidth), 1,6)
                .addBottleneck((int) (64 * scaleWidth), 1,6)
                .addBottleneck((int) (64 * scaleWidth), 1,6)
                .addBottleneck((int) (96 * scaleWidth), 1,6)
                .addBottleneck((int) (96 * scaleWidth), 1,6)
                .addBottleneck((int) (96 * scaleWidth), 1,6)
                .addBottleneck((int) (160 * scaleWidth), 2,6)
                .addBottleneck((int) (160 * scaleWidth), 1,6)
                .addBottleneck((int) (160 * scaleWidth), 1,6)
                .addBottleneck((int) (320 * scaleWidth), 1,6)
                .addConvolutionLayer((int) (1280 * scaleWidth), 1, 1)
                .addGlobalAveragePoolingLayer()
                .addDropoutLayer(dropout)
                .addDenseLayer(sizeOutput, new FunctionActivation.Softmax());

        return createMobilenet();
    }
}
