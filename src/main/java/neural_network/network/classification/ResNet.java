package neural_network.network.classification;

import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.layers.NeuralLayer;
import neural_network.layers.convolution_3d.*;
import neural_network.layers.convolution_3d.residual.ResidualBlock;
import neural_network.layers.convolution_3d.residual.ResidualUnit;
import neural_network.layers.dense.ActivationLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.layers.reshape.Flatten3DLayer;
import neural_network.layers.reshape.GlobalAveragePooling3DLayer;
import neural_network.network.NeuralNetwork;

public class ResNet {
    private final NeuralNetwork resnet;

    public ResNet() {
        this.resnet = new NeuralNetwork();
    }

    public ResNet addInputLayer(int height, int width, int depth) {
        resnet.addInputLayer(height, width, depth);

        return this;
    }

    public ResNet addLayer(NeuralLayer layer) {
        resnet.addLayer(layer);

        return this;
    }

    public ResNet addConvolutionLayer(int countKernel, int sizeKernel) {
        return addConvolutionLayer(countKernel, sizeKernel, 1, sizeKernel / 2);
    }

    public ResNet addConvolutionLayer(int countKernel, int sizeKernel, int stride, int padding) {
        resnet.addLayer(new ConvolutionLayer(countKernel, sizeKernel, stride, padding).setInitializer(new Initializer.HeNormal()));
        resnet.addLayer(new BatchNormalizationLayer3D(0.9));
        resnet.addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()));

        return this;
    }

    public ResNet addMaxPoolingLayer(int sizeKernel, int stride) {
        resnet.addLayer(new MaxPoolingLayer(sizeKernel, stride, sizeKernel / 2));

        return this;
    }

    public ResNet addGlobalAveragePoolingLayer() {
        resnet.addLayer(new GlobalAveragePooling3DLayer());
        return this;
    }

    public ResNet addDenseLayer(int countNeuron, FunctionActivation functionActivation) {
        if (resnet.getOutputSize().length != 1) {
            resnet.addLayer(new Flatten3DLayer());
        }
        resnet.addLayer(new DenseLayer(countNeuron).setInitializer(new Initializer.XavierNormal()));
        resnet.addLayer(new ActivationLayer(functionActivation));

        return this;
    }

    public ResNet addResBlock(int countKernel) {
        return addResBlock(countKernel, false);
    }

    public ResNet addResBlock(int countKernel, boolean downsample) {
        if (downsample) {
            return addDownsampleResBlock(countKernel);
        }
        return addResBlockV1(countKernel);
    }

    public ResNet addResBlockV2(int countKernel) {
        return addResBlockV2(countKernel, false);
    }

    public ResNet addResBlockV2(int countKernel, boolean downsample) {
        if (downsample) {
            return addDownsampleResBlockV2(countKernel);
        }
        return addResBlockPreActivation(countKernel);
    }

    public ResNet addBottleneckResBlock(int countKernel) {
        return addBottleneckResBlock(countKernel, false);
    }

    public ResNet addBottleneckResBlock(int countKernel, boolean downsample) {
        return addBottleneckResBlock(countKernel, 4, 4, downsample);
    }

    public ResNet addBottleneckResBlock(int countKernel, int scale1x1, int scale3x3, boolean downsample) {
        if (downsample) {
            return addDownsampleBottleneckResBlock(countKernel, scale1x1, scale3x3);
        }
        return addBottleneckResBlock(countKernel, scale1x1, scale3x3);
    }

    public ResNet addBottleneckResBlockV2(int countKernel) {
        return addBottleneckResBlockV2(countKernel, false);
    }

    public ResNet addBottleneckResBlockV2(int countKernel, boolean downsample) {
        return addBottleneckResBlockV2(countKernel, 4, 4, downsample);
    }

    public ResNet addBottleneckResBlockV2(int countKernel, int scale1x1, int scale3x3, boolean downsample) {
        if (downsample) {
            return addDownsampleBottleneckResBlockV2(countKernel, scale1x1, scale3x3);
        }
        return addBottleneckResBlockV2(countKernel, scale1x1, scale3x3);
    }

    public ResNet addResBlock(int countKernel, int countGroup) {
        return addResBlock(countKernel, countGroup, false);
    }

    public ResNet addResBlock(int countKernel, int countGroup, boolean downsample) {
        if (downsample) {
            return addDownsampleResBlock(countKernel, countGroup);
        }
        return addResBlockV1(countKernel, countGroup);
    }

    public ResNet addResBlockV2(int countKernel, int countGroup) {
        return addResBlockV2(countKernel, countGroup, false);
    }

    public ResNet addResBlockV2(int countKernel, int countGroup, boolean downsample) {
        if (downsample) {
            return addDownsampleResBlockV2(countKernel, countGroup);
        }
        return addResBlockPreActivation(countKernel, countGroup);
    }

    public ResNet addBottleneckResBlock(int countKernel, int countGroup) {
        return addBottleneckResBlock(countKernel, countGroup, false);
    }

    public ResNet addBottleneckResBlock(int countKernel, int countGroup, boolean downsample) {
        return addBottleneckResBlock(countKernel, 4, 4, countGroup, downsample);
    }

    public ResNet addBottleneckResBlock(int countKernel, int scale1x1, int scale3x3, int countGroup, boolean downsample) {
        if (downsample) {
            return addDownsampleBottleneckResBlock(countKernel, scale1x1, scale3x3, countGroup);
        }
        return addBottleneckResBlock(countKernel, scale1x1, scale3x3, countGroup);
    }

    public ResNet addBottleneckResBlockV2(int countKernel, int countGroup) {
        return addBottleneckResBlockV2(countKernel, countGroup, false);
    }

    public ResNet addBottleneckResBlockV2(int countKernel, int countGroup, boolean downsample) {
        return addBottleneckResBlockV2(countKernel, 4, 4, countGroup, downsample);
    }

    public ResNet addBottleneckResBlockV2(int countKernel, int scale1x1, int scale3x3, int countGroup, boolean downsample) {
        if (downsample) {
            return addDownsampleBottleneckResBlockV2(countKernel, scale1x1, scale3x3, countGroup);
        }
        return addBottleneckResBlockV2(countKernel, scale1x1, scale3x3, countGroup);
    }

    private ResNet addResBlockV1(int countKernel) {
        resnet.addLayer(
                new ResidualBlock()
                        .addResidualUnit(new ResidualUnit())
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new ConvolutionLayer(countKernel, 3, 1, 1))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel, 3, 1, 1))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                        )
        );
        resnet.addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()));

        return this;
    }

    private ResNet addBottleneckResBlock(int countKernel, int scale1x1, int scale3x3) {
        resnet.addLayer(
                new ResidualBlock()
                        .addResidualUnit(new ResidualUnit())
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new ConvolutionLayer(countKernel / scale1x1, 1, 1, 0))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel / scale3x3, 3, 1, 1))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel, 1, 1, 0))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                        )
        );
        resnet.addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()));

        return this;
    }

    private ResNet addDownsampleBottleneckResBlock(int countKernel, int scale1x1, int scale3x3) {
        resnet.addLayer(
                new ResidualBlock()
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new ConvolutionLayer(countKernel, 3, 2, 1))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                        )
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new ConvolutionLayer(countKernel / scale1x1, 1, 1, 0))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel / scale3x3, 3, 2, 1))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel, 1, 1, 0))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                        )
        );
        resnet.addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()));

        return this;
    }

    public ResNet addDownsampleResBlock(int countKernel) {
        resnet.addLayer(
                new ResidualBlock()
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new ConvolutionLayer(countKernel, 3,2, 1))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                        )
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new ConvolutionLayer(countKernel, 3, 2, 1))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel, 3, 1, 1))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                        )
        );
        resnet.addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()));

        return this;
    }

    private ResNet addResBlockPreActivation(int countKernel) {
        resnet.addLayer(
                new ResidualBlock()
                        .addResidualUnit(new ResidualUnit())
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel, 3, 1, 1))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel, 3, 1, 1))
                        )
        );

        return this;
    }

    private ResNet addBottleneckResBlockV2(int countKernel, int scale1x1, int scale3x3) {
        resnet.addLayer(
                new ResidualBlock()
                        .addResidualUnit(new ResidualUnit())
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel / scale1x1, 1, 1, 0))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel / scale3x3, 3, 1, 1))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel, 1, 1, 0))
                        )
        );

        return this;
    }

    private ResNet addDownsampleBottleneckResBlockV2(int countKernel, int scale1x1, int scale3x3) {
        resnet.addLayer(
                new ResidualBlock()
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel, 3,2, 1))
                        )
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel / scale1x1, 1, 1, 0))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel / scale3x3, 3, 2, 1))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel, 1, 1, 0))
                        )
        );

        return this;
    }

    public ResNet addDownsampleResBlockV2(int countKernel) {
        resnet.addLayer(
                new ResidualBlock()
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel, 3,2, 1))
                        )
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel, 3, 2, 1))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new ConvolutionLayer(countKernel, 3, 1, 1))
                        )
        );

        return this;
    }

    private ResNet addResBlockV1(int countKernel, int countGroup) {
        resnet.addLayer(
                new ResidualBlock()
                        .addResidualUnit(new ResidualUnit())
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new GroupedConvolutionLayer(countKernel, 3, 1, 1, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel, 3, 1, 1, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                        )
        );
        resnet.addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()));

        return this;
    }

    private ResNet addBottleneckResBlock(int countKernel, int scale1x1, int scale3x3, int countGroup) {
        resnet.addLayer(
                new ResidualBlock()
                        .addResidualUnit(new ResidualUnit())
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new GroupedConvolutionLayer(countKernel / scale1x1, 1, 1, 0, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel / scale3x3, 3, 2, 1, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel, 1, 1, 0, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                        )
        );
        resnet.addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()));

        return this;
    }

    private ResNet addDownsampleBottleneckResBlock(int countKernel, int scale1x1, int scale3x3, int countGroup) {
        resnet.addLayer(
                new ResidualBlock()
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new GroupedConvolutionLayer(countKernel, 3, 2, 1, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                        )
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new GroupedConvolutionLayer(countKernel / scale1x1, 1, 1, 0, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel / scale3x3, 3, 2, 1, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel, 1, 1, 0, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                        )
        );
        resnet.addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()));

        return this;
    }

    public ResNet addDownsampleResBlock(int countKernel, int countGroup) {
        resnet.addLayer(
                new ResidualBlock()
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new GroupedConvolutionLayer(countKernel, 3, 2, 1, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                        )
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new GroupedConvolutionLayer(countKernel, 3, 2, 1, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel, 3, 1, 1, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                        )
        );
        resnet.addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()));

        return this;
    }

    private ResNet addResBlockPreActivation(int countKernel, int countGroup) {
        resnet.addLayer(
                new ResidualBlock()
                        .addResidualUnit(new ResidualUnit())
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel, 3, 1, 1, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel, 3, 1, 1, countGroup))
                        )
        );

        return this;
    }

    private ResNet addBottleneckResBlockV2(int countKernel, int scale1x1, int scale3x3, int countGroup) {
        resnet.addLayer(
                new ResidualBlock()
                        .addResidualUnit(new ResidualUnit())
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel / scale1x1, 1, 1, 0, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel / scale3x3, 3, 2, 1, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel, 1, 1, 0, countGroup))
                        )
        );

        return this;
    }

    private ResNet addDownsampleBottleneckResBlockV2(int countKernel, int scale1x1, int scale3x3, int countGroup) {
        resnet.addLayer(
                new ResidualBlock()
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel, 3, 2, 1, countGroup))
                        )
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel / scale1x1, 1, 1, 0, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel / scale3x3, 3, 2, 1, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel, 1, 1, 0, countGroup))
                        )
        );

        return this;
    }

    public ResNet addDownsampleResBlockV2(int countKernel, int countGroup) {
        resnet.addLayer(
                new ResidualBlock()
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel, 3, 2, 1, countGroup))
                        )
                        .addResidualUnit(new ResidualUnit()
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel, 3, 2, 1, countGroup))
                                .addLayer(new BatchNormalizationLayer3D(0.9))
                                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                                .addLayer(new GroupedConvolutionLayer(countKernel, 3, 1, 1, countGroup))
                        )
        );

        return this;
    }

    public NeuralNetwork createResNet() {
        return resnet;
    }
}