package neural_network.layers;

import neural_network.layers.convolution_2d.ActivationLayer2D;
import neural_network.layers.convolution_2d.DropoutLayer2D;
import neural_network.layers.convolution_3d.*;
import neural_network.layers.convolution_3d.densely.DenseModule;
import neural_network.layers.convolution_3d.inception.InceptionModule;
import neural_network.layers.convolution_3d.residual.ResidualModule;
import neural_network.layers.convolution_3d.squeeze_and_excitation.SEBlock;
import neural_network.layers.convolution_3d.u_net.UConcatenateLayer;
import neural_network.layers.dense.*;
import neural_network.layers.recurrent.*;
import neural_network.layers.reshape.*;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public abstract class NeuralLayer {
    protected boolean trainable;
    protected ArrayList<NeuralLayer> nextLayers;

    public NeuralLayer(){
        nextLayers = new ArrayList<>();
    }

    public static void read(Scanner scanner, ArrayList<NeuralLayer> layers){
        String layer = scanner.nextLine();
        while (!layer.equals("End")) {
            switch (layer) {
                case "Dense layer" -> layers.add(DenseLayer.read(scanner));
                case "Spectral normalization dense layer" -> layers.add(SNDenseLayer.read(scanner));
                case "Variational layer" -> layers.add(VariationalLayer.read(scanner));
                case "Dropout layer" -> layers.add(DropoutLayer.read(scanner));
                case "Activation layer" -> layers.add(ActivationLayer.read(scanner));
                case "Activation layer 3D" -> layers.add(ActivationLayer3D.read(scanner));
                case "Activation layer 2D" -> layers.add(ActivationLayer2D.read(scanner));
                case "Average pooling layer 3D" -> layers.add(AveragePoolingLayer.read(scanner));
                case "Batch normalization layer 3D" -> layers.add(BatchNormalizationLayer3D.read(scanner));
                case "Instance normalization layer 3D" -> layers.add(InstanceNormalizationLayer3D.read(scanner));
                case "Batch renormalization layer 3D" -> layers.add(BatchRenormalizationLayer3D.read(scanner));
                case "Convolution layer 3D" -> layers.add(ConvolutionLayer.read(scanner));
                case "Shuffled layer" -> layers.add(ShuffledLayer.read(scanner));
                case "Spectral normalization convolution layer 3D" -> layers.add(SNConvolutionLayer.read(scanner));
                case "Depthwise convolution layer 3D" -> layers.add(DepthwiseConvolutionLayer.read(scanner));
                case "Dilated convolution layer 3D" -> layers.add(DilatedConvolutionLayer.read(scanner));
                case "Grouped convolution layer 3D" -> layers.add(GroupedConvolutionLayer.read(scanner));
                case "Convolution layer 2D" -> layers.add(neural_network.layers.convolution_2d.ConvolutionLayer.read(scanner));
                case "Convolution transpose layer 3D" -> layers.add(ConvolutionTransposeLayer.read(scanner));
                case "Spectral normalization convolution transpose layer 3D" -> layers.add(SNConvolutionTransposeLayer.read(scanner));
                case "Dropout layer 3D" -> layers.add(DropoutLayer3D.read(scanner));
                case "Dropout layer 2D" -> layers.add(DropoutLayer2D.read(scanner));
                case "Max pooling layer 3D" -> layers.add(MaxPoolingLayer.read(scanner));
                case "Up sampling layer" -> layers.add(UpSamplingLayer.read(scanner));
                case "Flatten layer 3D" -> layers.add(Flatten3DLayer.read(scanner));
                case "Flatten layer 2D" -> layers.add(Flatten2DLayer.read(scanner));
                case "Global max pooling 3D" -> layers.add(GlobalMaxPooling3DLayer.read(scanner));
                case "Global max pooling 2D" -> layers.add(GlobalMaxPooling2DLayer.read(scanner));
                case "Global average pooling 3D" -> layers.add(GlobalAveragePooling3DLayer.read(scanner));
                case "Global average pooling 2D" -> layers.add(GlobalAveragePooling2DLayer.read(scanner));
                case "Reshape layer 3D" -> layers.add(Reshape3DLayer.read(scanner));
                case "Inception module" -> layers.add(InceptionModule.read(scanner));
                case "SE block" -> layers.add(SEBlock.read(scanner));
                case "Layers block" -> layers.add(LayersBlock.readBlock(scanner));
                case "U concatenate layer" -> layers.add(UConcatenateLayer.read(layers, scanner));
                case "Batch normalization layer" -> layers.add(BatchNormalizationLayer.read(scanner));
                case "Batch renormalization layer" -> layers.add(BatchRenormalizationLayer.read(scanner));
                case "LSTM layer" -> layers.add(LSTMLayer.read(scanner));
                case "Peephole LSTM layer" -> layers.add(PeepholeLSTMLayer.read(scanner));
                case "GRU layer" -> layers.add(GRULayer.read(scanner));
                case "Recurrent layer" -> layers.add(RecurrentLayer.read(scanner));
                case "Bidirectional block" -> layers.add(Bidirectional.read(scanner));
                case "Residual module" -> layers.add(ResidualModule.read(scanner));
                case "Dense module" -> layers.add(DenseModule.read(scanner));
                case "Embedding layer" -> layers.add(EmbeddingLayer.read(scanner));
            }
            layer = scanner.nextLine();
        }
    }

    public abstract int[] size();

    public abstract void initialize(Optimizer optimizer);

    public abstract int info();

    public abstract void write(FileWriter writer) throws IOException;

    public abstract void initialize(int[] size);

    public abstract void generateOutput(NNArray[] input);

    public abstract void generateTrainOutput(NNArray[] input);

    public abstract void generateError(NNArray[] error);

    public abstract NNArray[] getOutput();

    public abstract NNArray[] getError();

    public NNArray[] getErrorNL(){
        return getError();
    };

    public void addNextLayer(NeuralLayer neuralLayer){
        nextLayers.add(neuralLayer);
    }

    public void trainable(boolean trainable){
        this.trainable = trainable;
    }

    protected int getCountCores(){
        return Math.min(Runtime.getRuntime().availableProcessors() + 2, getOutput().length);
    }
}
