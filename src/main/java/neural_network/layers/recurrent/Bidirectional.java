package neural_network.layers.recurrent;

import lombok.NoArgsConstructor;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

@NoArgsConstructor
public class Bidirectional extends RecurrentNeuralLayer {
    private RecurrentNeuralLayer forwardLayer;
    private RecurrentNeuralLayer backLayer;

    public Bidirectional(RecurrentLayer forwardLayer) {
        this.forwardLayer = forwardLayer;
        this.backLayer = new RecurrentLayer(forwardLayer);
    }

    public Bidirectional(GRULayer forwardLayer) {
        this.forwardLayer = forwardLayer;
        this.backLayer = new GRULayer(forwardLayer);
    }

    public Bidirectional(LSTMLayer forwardLayer) {
        this.forwardLayer = forwardLayer;
        this.backLayer = new LSTMLayer(forwardLayer);
    }

    public Bidirectional(PeepholeLSTMLayer forwardLayer) {
        this.forwardLayer = forwardLayer;
        this.backLayer = new PeepholeLSTMLayer(forwardLayer);
    }

    @Override
    public void initialize(int[] size) {
        if (size.length != 2) {
            throw new ExceptionInInitializerError("Error size pre layer!");
        }

        forwardLayer.initialize(size);
        backLayer.initialize(size);

        forwardLayer.returnState = this.returnState;
        backLayer.returnState = this.returnState;

        width = size[0];
        depth = size[1];
        if (returnSequences) {
            outWidth = width;
        } else {
            outWidth = 1;
        }
        outDepth = forwardLayer.countNeuron * 2;
    }

    @Override
    public void initialize(Optimizer optimizer) {
        forwardLayer.initialize(optimizer);
        backLayer.initialize(optimizer);
    }

    public Bidirectional setPreLayer(RecurrentNeuralLayer layer) {
        super.setPreLayer(layer);

        forwardLayer.preLayer = layer;
        backLayer.preLayer = layer;

        return this;
    }

    @Override
    public int info() {
        System.out.println("            |      Bidirectional block      |             ");
        System.out.println("____________|_______________________________|_____________");
        int countParam =  forwardLayer.info();
        System.out.println("____________|_______________|_______________|_____________");
        countParam += backLayer.info();
        System.out.println("____________|_______________|_______________|_____________");
        System.out.println("            |  " + width + ",\t" + depth + "\t\t|  " + outWidth + ",\t" + outDepth + "\t\t|\t" + countParam);

        return countParam;
    }

    @Override
    public void write(FileWriter writer) throws IOException {
        writer.write("Bidirectional block\n");
        writer.write(returnState + "\n");
        forwardLayer.write(writer);
        backLayer.write(writer);
        writer.write(trainable + "\n");
        writer.flush();
    }

    public static Bidirectional read(Scanner scanner){
        Bidirectional bidirectional = new Bidirectional();
        bidirectional.returnState = Boolean.parseBoolean(scanner.nextLine());

        bidirectional.forwardLayer = readRecurrentLayer(scanner);
        bidirectional.backLayer = readRecurrentLayer(scanner);

        bidirectional.trainable = Boolean.parseBoolean(scanner.nextLine());
        return bidirectional;
    }

    private static RecurrentNeuralLayer readRecurrentLayer(Scanner scanner){
        String layer = scanner.nextLine();
        RecurrentNeuralLayer recurrentLayer = null;
        switch (layer) {
            case "LSTM layer" -> recurrentLayer = LSTMLayer.read(scanner);
            case "Peephole LSTM layer" -> recurrentLayer = PeepholeLSTMLayer.read(scanner);
            case "GRU layer" -> recurrentLayer = GRULayer.read(scanner);
            case "Recurrent layer" -> recurrentLayer = RecurrentLayer.read(scanner);
        }
        return recurrentLayer;
    }

    @Override
    public void generateOutput(NNArray[] input) {
        this.input = NNArrays.isMatrix(input);
        forwardLayer.generateOutput(input);
        backLayer.generateOutput(NNArrays.reverse(this.input));
        this.output = NNArrays.concatMatrix(forwardLayer.getOutput(), backLayer.getOutput());

        if(returnState){
            generateState();
        }
    }

    private void generateState(){
        state = new NNVector[input.length][];
        for (int i = 0; i < input.length; i++) {
            state[i] = NNArrays.concatVector(forwardLayer.errorState[i], backLayer.errorState[i]);
        }
    }

    @Override
    public void generateTrainOutput(NNArray[] inputs) {
        forwardLayer.dropout = true;
        backLayer.dropout = true;
        generateOutput(inputs);
        forwardLayer.dropout = false;
        backLayer.dropout = false;
    }

    @Override
    public void trainable(boolean trainable){
        this.trainable = trainable;
        forwardLayer.trainable(trainable);
        backLayer.trainable(trainable);
    }

    @Override
    public void generateError(NNArray[] error) {
        NNArray[] errorFL = NNArrays.subArray(error, forwardLayer.getOutput());
        NNArray[] errorBL = NNArrays.subArray(error, backLayer.getOutput(), forwardLayer.getOutput()[0].size());

        forwardLayer.generateError(errorFL);
        backLayer.generateError(errorBL);

        this.error = NNArrays.create(this.input);
        NNArrays.add(this.error, forwardLayer.getError());
        NNArrays.add(this.error, backLayer.getError());

        if(hasPreLayer()) {
            generateErrorState();
        }
    }

    private void generateErrorState(){
        this.errorState = new NNVector[input.length][];
        for (int i = 0; i < errorState.length; i++) {
            this.errorState[i] = new NNVector[forwardLayer.errorState[i].length];
            for (int j = 0; j < errorState[i].length; j++) {
                this.errorState[i][j] = new NNVector(forwardLayer.countNeuron);

                this.errorState[i][j].add(forwardLayer.errorState[i][j]);
                this.errorState[i][j].add(backLayer.errorState[i][j]);
            }
        }
    }
}
