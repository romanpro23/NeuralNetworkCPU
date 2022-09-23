package neural_network.layers;

import lombok.Setter;
import neural_network.layers.dense.*;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public abstract class NeuralLayer {
    protected boolean trainable;
    protected ArrayList<NeuralLayer> preLayers;
    protected ArrayList<NeuralLayer> nextLayers;

    public NeuralLayer(){
        preLayers = new ArrayList<>();
        nextLayers = new ArrayList<>();
    }

    public static void read(Scanner scanner, ArrayList<NeuralLayer> layers){
        String layer = scanner.nextLine();
        while (!layer.equals("End")) {
            switch (layer) {
                case "Dense layer" -> layers.add(DenseLayer.read(scanner));
//                case "Variational layer" -> layers.add(VariationalLayer.read(scanner));
                case "Dropout layer" -> layers.add(DropoutLayer.read(scanner));
                case "Activation layer" -> layers.add(ActivationLayer.read(scanner));
                case "Batch normalization layer" -> layers.add(BatchNormalizationLayer.read(scanner));
                case "Batch renormalization layer" -> layers.add(BatchRenormalizationLayer.read(scanner));
            }

            layer = scanner.nextLine();
        }
    }

    public abstract int[] size();

    public abstract void initialize(Optimizer optimizer);

    public abstract void update(Optimizer optimizer);

    public abstract int info();

    public abstract void write(FileWriter writer) throws IOException;

    public abstract void initialize(int[] size);

    public abstract void generateOutput(NNArray[] input);

    public abstract void generateTrainOutput(NNArray[] input);

    public abstract void generateError(NNArray[] error);

    public abstract NNArray[] getOutput();

    public abstract NNArray[] getError();

    public void addPreLayer(NeuralLayer neuralLayer){
        preLayers.add(neuralLayer);
    }

    public void addNextLayer(NeuralLayer neuralLayer){
        nextLayers.add(neuralLayer);
    }

    public void trainable(boolean trainable){
        this.trainable = trainable;
    }
}
