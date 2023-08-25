package neural_network.layers.recurrent;

import lombok.NoArgsConstructor;
import neural_network.optimizers.Optimizer;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNVector;
import utilities.CublasUtil;

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

//    public Bidirectional(GRUBahdAttentionLayer forwardLayer) {
//        this.forwardLayer = forwardLayer;
//        this.backLayer = new GRUBahdAttentionLayer(forwardLayer);
//    }
//
//    public Bidirectional(GRULuongAttentionLayer forwardLayer) {
//        this.forwardLayer = forwardLayer;
//        this.backLayer = new GRULuongAttentionLayer(forwardLayer);
//    }
//
//    public Bidirectional(LSTMLuongAttentionLayer forwardLayer) {
//        this.forwardLayer = forwardLayer;
//        this.backLayer = new LSTMLuongAttentionLayer(forwardLayer);
//    }
//
//    public Bidirectional(LSTMBahdAttentionLayer forwardLayer) {
//        this.forwardLayer = forwardLayer;
//        this.backLayer = new LSTMBahdAttentionLayer(forwardLayer);
//    }

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
    public void generateOutput(CublasUtil.Matrix[] input_gpu) {

    }

    @Override
    public void initialize(Optimizer optimizer) {
        forwardLayer.initialize(optimizer);
        backLayer.initialize(optimizer);
    }

    @Override
    public int info() {
        System.out.println("            |      Bidirectional block      |             ");
        System.out.println("____________|_______________________________|_____________");
        int countParam = forwardLayer.info();
        System.out.println("____________|_______________|_______________|_____________");
        countParam += backLayer.info();
        System.out.println("____________|_______________|_______________|_____________");
        System.out.println("            |  " + width + ",\t" + depth + "\t\t|  " + outWidth + ",\t" + outDepth + "\t\t|\t" + countParam);

        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Bidirectional block\n");
        forwardLayer.save(writer);
        backLayer.save(writer);
        writer.write(trainable + "\n");
        writer.flush();
    }

    @Override
    public Bidirectional setReturnSequences(boolean returnSequences) {
        this.forwardLayer.setReturnSequences(returnSequences);
        this.backLayer.setReturnSequences(returnSequences);

        return this;
    }

    @Override
    public void generateOutput(NNArray[] input, NNArray[][] state) {
        this.input = NNArrays.isMatrix(input);
        forwardLayer.generateOutput(input, state);
        backLayer.generateOutput(NNArrays.reverse(this.input), state);

        this.output = NNArrays.concatMatrix(forwardLayer.getOutput(), backLayer.getOutput());
        this.state = new NNVector[input.length][];
        for (int i = 0; i < input.length; i++) {
            this.state[i] = NNArrays.concatVector(forwardLayer.getState()[i], backLayer.getState()[i]);
        }
    }

    @Override
    public void generateError(NNArray[] error, NNArray[][] errorState) {
        NNMatrix[] errorFL = NNArrays.subMatrix(error, forwardLayer.getOutput(), 0);
        NNMatrix[] errorBL = NNArrays.subMatrix(error, backLayer.getOutput(), forwardLayer.getOutput()[0].shape()[1]);

        NNVector[][] errorStateFL = new NNVector[input.length][];
        NNVector[][] errorStateBL = new NNVector[input.length][];
        if(errorState != null) {
            for (int i = 0; i < errorState.length; i++) {
                errorStateFL[i] = new NNVector[forwardLayer.state[i].length];
                errorStateBL[i] = new NNVector[backLayer.state[i].length];
                for (int j = 0; j < errorState[i].length; j++) {
                    errorStateFL[i][j] = errorState[i][j].subVector(0, forwardLayer.countNeuron);
                    errorStateBL[i][j] = errorState[i][j].subVector(forwardLayer.countNeuron, backLayer.countNeuron);
                }
            }
        }

        forwardLayer.generateError(errorFL, errorStateFL);
        backLayer.generateError(errorBL, errorStateBL);

        this.error = NNArrays.create(this.input);
        NNArrays.add(this.error, forwardLayer.getError());
        NNArrays.add(this.error, backLayer.getError());

        errorState = new NNVector[input.length][];
        for (int i = 0; i < errorState.length; i++) {
            errorState[i] = new NNVector[forwardLayer.getState()[i].length];
            for (int j = 0; j < errorState[i].length; j++) {
                errorState[i][j] = new NNVector(forwardLayer.countNeuron);
                errorState[i][j].add(forwardLayer.getErrorState()[i][j]);
                errorState[i][j].add(backLayer.getErrorState()[i][j]);
            }
        }
    }

    public static Bidirectional read(Scanner scanner) {
        Bidirectional bidirectional = new Bidirectional();

        bidirectional.forwardLayer = readRecurrentLayer(scanner);
        bidirectional.backLayer = readRecurrentLayer(scanner);

        bidirectional.trainable = Boolean.parseBoolean(scanner.nextLine());
        return bidirectional;
    }

    private static RecurrentNeuralLayer readRecurrentLayer(Scanner scanner) {
        String layer = scanner.nextLine();
        RecurrentNeuralLayer recurrentLayer = null;
        switch (layer) {
            case "LSTM layer" -> recurrentLayer = LSTMLayer.read(scanner);
            case "Peephole LSTM layer" -> recurrentLayer = PeepholeLSTMLayer.read(scanner);
            case "GRU layer" -> recurrentLayer = GRULayer.read(scanner);
            case "Recurrent layer" -> recurrentLayer = RecurrentLayer.read(scanner);
//            case "GRU luong attention layer" -> recurrentLayer = (GRULuongAttentionLayer.read(scanner));
//            case "GRU bahdanau attention layer" -> recurrentLayer = (GRUBahdAttentionLayer.read(scanner));
//            case "LSTM luong attention layer" -> recurrentLayer = (LSTMLuongAttentionLayer.read(scanner));
//            case "LSTM bahdanau attention layer" -> recurrentLayer = (LSTMBahdAttentionLayer.read(scanner));
        }
        return recurrentLayer;
    }

    @Override
    public void generateTrainOutput(NNArray[] inputs, NNArray[][] state) {
        forwardLayer.dropout = true;
        backLayer.dropout = true;
        generateOutput(inputs, state);
        forwardLayer.dropout = false;
        backLayer.dropout = false;
    }

    @Override
    public void generateTrainOutput(NNArray[] inputs) {
        forwardLayer.dropout = true;
        backLayer.dropout = true;
        generateOutput(inputs, null);
        forwardLayer.dropout = false;
        backLayer.dropout = false;
    }

    @Override
    public void generateError(CublasUtil.Matrix[] errors) {

    }

    @Override
    public void trainable(boolean trainable) {
        this.trainable = trainable;
        forwardLayer.trainable(trainable);
        backLayer.trainable(trainable);
    }
}
