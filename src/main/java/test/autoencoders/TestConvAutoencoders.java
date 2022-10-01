package test.autoencoders;

import data.ImageCreator;
import data.mnist.MNISTLoader1D;
import data.mnist.MNISTLoader3D;
import neural_network.activation.FunctionActivation;
import neural_network.layers.convolution_3d.ActivationLayer3D;
import neural_network.layers.convolution_3d.BatchNormalizationLayer3D;
import neural_network.layers.convolution_3d.ConvolutionLayer;
import neural_network.layers.convolution_3d.ConvolutionTransposeLayer;
import neural_network.layers.dense.DenseLayer;
import neural_network.layers.reshape.Flatten3DLayer;
import neural_network.layers.reshape.Reshape3DLayer;
import neural_network.loss.FunctionLoss;
import neural_network.network.NeuralNetwork;
import neural_network.network.autoencoders.Autoencoder;
import neural_network.optimizers.AdamOptimizer;
import nnarrays.NNArrays;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class TestConvAutoencoders {
    public static void main(String[] args) throws IOException {
        NeuralNetwork encoder = new NeuralNetwork()
                .addInputLayer(28, 28, 1)
                .addLayer(new ConvolutionLayer(16, 3, 2, 1))
                .addLayer(new BatchNormalizationLayer3D(0.9))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new ConvolutionLayer(32, 3, 2, 1))
                .addLayer(new BatchNormalizationLayer3D(0.9))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new Flatten3DLayer())
                .addDenseLayer(256)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DenseLayer(32))
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.Quadratic())
                .create();

        NeuralNetwork decoder = new NeuralNetwork()
                .addInputLayer(32)
                .addDenseLayer(128)
                .addActivationLayer(new FunctionActivation.ReLU())
                .addDenseLayer(7*7*32)
                .addLayer(new Reshape3DLayer(7, 7, 32))
                .addLayer(new ConvolutionTransposeLayer(32, 3, 2, 1))
                .addLayer(new BatchNormalizationLayer3D(0.9))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new ConvolutionTransposeLayer(16, 3, 2, 1))
                .addLayer(new BatchNormalizationLayer3D(0.9))
                .addLayer(new ActivationLayer3D(new FunctionActivation.ReLU()))
                .addLayer(new ConvolutionLayer(1, 3, 1, 1))
                .addLayer(new ActivationLayer3D(new FunctionActivation.Sigmoid()))
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.Quadratic())
                .create();

        Autoencoder autoencoder = new Autoencoder(encoder, decoder);
        MNISTLoader3D loader = new MNISTLoader3D();

        for (int i = 0; i < 100000; i++) {
            if (i % 25 == 0) {
                NNTensor[] data = NNArrays.isTensor(loader.getNextTestData(1).getInput());
                ImageCreator.drawImage(data[0], 28, 28, i / 25 + "_input", "D:/NetworkTest/CAE");
                NNTensor result = NNArrays.toTensor(autoencoder.query(data), 28, 28, 1)[0];
                ImageCreator.drawImage(result, 28, 28, i / 25 + "_output", "D:/NetworkTest/CAE");
                encoder.save(new FileWriter(new File("D:/NetworkTest/CAE/encoder_32h.txt")));
                decoder.save(new FileWriter(new File("D:/NetworkTest/CAE/decoder_32h.txt")));
            }
            System.out.println(i + " - " + autoencoder.train(loader.getNextTrainData(32).getInput()));
        }
    }
}
