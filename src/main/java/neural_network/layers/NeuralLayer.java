package neural_network.layers;

import jcuda.runtime.JCuda;
import lombok.Getter;
import neural_network.layers.capsule.*;
import neural_network.layers.layer_2d.*;
import neural_network.layers.layer_3d.*;
import neural_network.layers.layer_3d.ConvolutionLayer;
import neural_network.layers.layer_3d.attention.AttentionBlock;
import neural_network.layers.layer_3d.attention.ChannelAttentionModule;
import neural_network.layers.layer_3d.attention.SpatialAttentionModule;
import neural_network.layers.layer_3d.densely.DenseBlock;
import neural_network.layers.layer_3d.densely.ResidualDenseBlock;
import neural_network.layers.layer_3d.inception.InceptionBlock;
import neural_network.layers.layer_3d.residual.ResidualBlock;
import neural_network.layers.layer_3d.attention.SEBlock;
import neural_network.layers.layer_3d.u_net.ConcatenateLayer;
import neural_network.layers.layer_1d.*;
import neural_network.layers.recurrent.*;
import neural_network.layers.reshape.*;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import utilities.Use;

import java.io.FileWriter;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import static utilities.GPUInit.allocated;
import static utilities.GPUInit.allocatedUse;

public abstract class NeuralLayer {
    @Getter
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
                case "Dense layer 2D" -> layers.add(DenseLayer2D.read(scanner));
                case "Mask layer" -> layers.add(MaskLayer.read(scanner));
                case "Image patches layer" -> layers.add(ImagePatchesLayer.read(scanner));
                case "VIT positional embedding layer" -> layers.add(VITPositionalEmbeddingLayer.read(scanner));
                case "Spectral normalization dense layer" -> layers.add(SNDenseLayer.read(scanner));
                case "Variational layer" -> layers.add(VariationalLayer.read(scanner));
                case "Dropout layer" -> layers.add(DropoutLayer.read(scanner));
                case "Activation layer" -> layers.add(ActivationLayer.read(scanner));
                case "Activation layer 2D" -> layers.add(ActivationLayer2D.read(scanner));
                case "Softmax layer 2D" -> layers.add(SoftmaxLayer2D.read(scanner));
                case "Activation layer 3D" -> layers.add(ActivationLayer3D.read(scanner));
                case "Parametric ReLU activation layer 3D" -> layers.add(ParametricReLULayer3D.read(scanner));
                case "Parametric ReLU activation layer" -> layers.add(ParametricReLULayer.read(scanner));
                case "Random ReLU activation layer 3D" -> layers.add(RandomReLULayer3D.read(scanner));
                case "Average pooling layer 3D" -> layers.add(AveragePoolingLayer.read(scanner));
                case "Batch normalization layer 3D" -> layers.add(BatchNormalizationLayer3D.read(scanner));
                case "Instance normalization layer 3D" -> layers.add(InstanceNormalizationLayer3D.read(scanner));
                case "Batch renormalization layer 3D" -> layers.add(BatchRenormalizationLayer3D.read(scanner));
                case "Convolution layer 3D" -> layers.add(ConvolutionLayer.read(scanner));
                case "Shuffled layer" -> layers.add(ShuffledLayer.read(scanner));
                case "Spectral normalization convolution layer 3D" -> layers.add(SNConvolutionLayer.read(scanner));
                case "Dilated convolution layer 3D" -> layers.add(DilatedConvolutionLayer.read(scanner));
                case "Grouped convolution layer 3D" -> layers.add(GroupedConvolutionLayer.read(scanner));
                case "Convolution layer 2D" -> layers.add(neural_network.layers.layer_2d.ConvolutionLayer.read(scanner));
                case "Convolution transpose layer 3D" -> layers.add(ConvolutionTransposeLayer.read(scanner));
                case "Spectral normalization convolution transpose layer 3D" -> layers.add(SNConvolutionTransposeLayer.read(scanner));
                case "Dropout layer 3D" -> layers.add(DropoutLayer3D.read(scanner));
                case "Dropout layer 2D" -> layers.add(DropoutLayer2D.read(scanner));
                case "Max pooling layer 3D" -> layers.add(MaxPoolingLayer.read(scanner));
                case "Up sampling layer" -> layers.add(UpSamplingLayer.read(scanner));
                case "Flatten layer 3D" -> layers.add(FlattenLayer3D.read(scanner));
                case "Flatten layer 2D" -> layers.add(FlattenLayer2D.read(scanner));
                case "Global max pooling 3D" -> layers.add(GlobalMaxPoolingLayer3D.read(scanner));
                case "Global max pooling 2D" -> layers.add(GlobalMaxPooling2DLayer.read(scanner));
                case "Global average pooling 3D" -> layers.add(GlobalAveragePoolingLayer3D.read(scanner));
                case "Global average pooling 2D" -> layers.add(GlobalAveragePooling2DLayer.read(scanner));
                case "Reshape layer 3D" -> layers.add(ReshapeLayer3D.read(scanner));
                case "Pixel shuffler layer 3D" -> layers.add(PixelShufflerLayer.read(scanner));
                case "Inception block" -> layers.add(InceptionBlock.read(scanner));
                case "SE block" -> layers.add(SEBlock.read(scanner));
                case "Attention block" -> layers.add(AttentionBlock.read(scanner));
                case "Channel attention module" -> layers.add(ChannelAttentionModule.read(scanner));
                case "Spatial attention module" -> layers.add(SpatialAttentionModule.read(scanner));
                case "Primary capsule layer" -> layers.add(PrimaryCapsuleLayer.read(scanner));
                case "Capsule layer" -> layers.add(CapsuleLayer.read(scanner));
                case "Digit capsule layer" -> layers.add(DigitCapsuleLayer.read(scanner));
                case "Squash activation layer" -> layers.add(SquashActivationLayer.read(scanner));
                case "Layers block" -> layers.add(LayersBlock.readBlock(scanner));
                case "Concatenate layer" -> layers.add(ConcatenateLayer.read(layers, scanner));
                case "U concatenate layer" -> layers.add(ConcatenateLayer.read(layers, scanner));
                case "Batch normalization layer" -> layers.add(BatchNormalizationLayer.read(scanner));
                case "Batch renormalization layer" -> layers.add(BatchRenormalizationLayer.read(scanner));
                case "LSTM layer" -> layers.add(LSTMLayer.read(scanner));
                case "Peephole LSTM layer" -> layers.add(PeepholeLSTMLayer.read(scanner));
                case "GRU layer" -> layers.add(GRULayer.read(scanner));
                case "GRU luong attention layer" -> layers.add(GRULuongAttentionLayer.read(scanner));
                case "GRU bahdanau attention layer" -> layers.add(GRUBahdAttentionLayer.read(scanner));
                case "Recurrent layer" -> layers.add(RecurrentLayer.read(scanner));
                case "Bidirectional block" -> layers.add(Bidirectional.read(scanner));
                case "Residual block" -> layers.add(ResidualBlock.read(scanner));
                case "Dense block" -> layers.add(DenseBlock.read(scanner));
                case "Residual dense block" -> layers.add(ResidualDenseBlock.read(scanner));
                case "Embedding layer" -> layers.add(EmbeddingLayer.read(scanner));
                case "Positional embedding layer" -> layers.add(PositionalEmbeddingLayer.read(scanner));
                case "Normalization layer" -> layers.add(NormalizationLayer.read(scanner));
                case "Normalization layer 2D" -> layers.add(NormalizationLayer2D.read(scanner));
                case "Additional block" -> layers.add(AdditionBlock.read(scanner));
                case "Multi head attention layer" -> layers.add(MultiHeadAttentionLayer.read(scanner));
                case "Embedding layer 3D" -> layers.add(EmbeddingLayer3D.read(scanner));
                case "Deformable convolution layer 3D" -> layers.add(DeformableConvolutionLayer.read(scanner));
                case "Modulated deformable convolution layer 3D" -> layers.add(DeformableV2ConvolutionLayer.read(scanner));
            }
            layer = scanner.nextLine();
        }
    }

    public abstract int[] size();

    public abstract void initialize(Optimizer optimizer);

    public abstract int info();

    public abstract void save(FileWriter writer) throws IOException;

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

    public void CallGarbageCollector()
    {
        System.gc();
        Runtime.getRuntime().gc();

        List<String> listString = new ArrayList<>();

        allocated.forEach((key, value) ->
        {
            Object arr = ((WeakReference<Object>) value).get();
            if (arr == null)
            {
                Use U = allocatedUse.get(key);
                if (U.data_gpu != null) JCuda.cudaFree(U.data_gpu);
                if (U.rowsIndex_gpu != null) JCuda.cudaFree(U.rowsIndex_gpu);
                if (U.columnsIndex_gpu != null) JCuda.cudaFree(U.columnsIndex_gpu);

                listString.add(key);
            }
        });

        listString.forEach((key) ->
        {
            allocated.remove(key);
            allocatedUse.remove(key);
        });

        listString.clear();
    }
}
