package neural_network.layers.convolution_3d.squeeze_and_excitation;

import neural_network.activation.FunctionActivation;
import neural_network.layers.LayersBlock;
import neural_network.layers.NeuralLayer;
import neural_network.layers.convolution_3d.UpSamplingLayer;
import neural_network.layers.dense.*;
import neural_network.layers.reshape.Flatten3DLayer;
import neural_network.layers.reshape.GlobalAveragePooling3DLayer;
import neural_network.layers.reshape.GlobalMaxPooling3DLayer;
import neural_network.layers.reshape.Reshape3DLayer;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class SEBlock extends LayersBlock {
    private int depth, height, width;
    private int outHeight, outWidth, outDepth;

    public SEBlock() {
        layers = new ArrayList<>();
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 3) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }

        height = size[0];
        width = size[1];
        depth = size[2];
        outWidth = width;
        outHeight = height;
        outDepth = depth;

        layers.add(new Reshape3DLayer(1, 1, depth));
        layers.add(new UpSamplingLayer(height, width));

        super.initialize(size);
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("SE block\n");
        for (NeuralLayer neuralLayer : layers) {
            neuralLayer.write(writer);
        }
        writer.write("End\n");
        writer.flush();
    }

    public static SEBlock read(Scanner scanner) {
        SEBlock seBlock = new SEBlock();
        NeuralLayer.read(scanner, seBlock.layers);

        return seBlock;
    }


    public SEBlock addGlobalPoolingLayer(Flatten3DLayer layer) {
        layers.add(layer);

        return this;
    }

    public SEBlock addDenseLayer(DenseLayer denseLayer) {
        layers.add(denseLayer);
        return this;
    }

    public SEBlock addDenseLayer(int countNeurons) {
        DenseLayer denseLayer = new DenseLayer(countNeurons);
        addDenseLayer(denseLayer);
        return this;
    }

    public SEBlock addDenseLayer(int countNeurons, FunctionActivation functionActivation) {
        DenseLayer denseLayer = new DenseLayer(countNeurons);
        addDenseLayer(denseLayer);
        addActivationLayer(functionActivation);
        return this;
    }

    public SEBlock addActivationLayer(FunctionActivation functionActivation) {
        addActivationLayer(new ActivationLayer(functionActivation));
        return this;
    }

    public SEBlock addActivationLayer(ActivationLayer activationLayer) {
        layers.add(activationLayer);
        return this;
    }

    public SEBlock addDropoutLayer(double chanceDropNeuron) {
        DropoutLayer dropoutLayer = new DropoutLayer(chanceDropNeuron);
        layers.add(dropoutLayer);
        return this;
    }

    public SEBlock addDropoutLayer(DropoutLayer dropoutLayer) {
        layers.add(dropoutLayer);
        return this;
    }

    public SEBlock addBatchNormalizationLayer(BatchNormalizationLayer batchNormalizationLayer) {
        layers.add(batchNormalizationLayer);
        return this;
    }

    public SEBlock addBatchNormalizationLayer() {
        return addBatchNormalizationLayer(new BatchNormalizationLayer());
    }

    public SEBlock addLayer(NeuralLayer layer) {
        layers.add(layer);
        return this;
    }

    @Override
    public int info() {
        int countParam = 0;
        System.out.println("            | Squeeze-and-Excitation layers |             ");
        System.out.println("____________|_______________________________|_____________");
        for (NeuralLayer neuralLayer : layers) {
            countParam += neuralLayer.info();
            System.out.println("____________|_______________|_______________|_____________");
        }
        System.out.println("            |  " + height + ",\t" + width + ",\t" + depth + "\t|  "
                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
        System.out.println("____________|_______________|_______________|_____________");
        return countParam;
    }

    public SEBlock setTrainable(boolean trainable) {
        super.setTrainable(trainable);

        return this;
    }
}
