package neural_network.layers.recurrent;

import neural_network.activation.FunctionActivation;
import neural_network.initialization.Initializer;
import neural_network.optimizers.Optimizer;
import neural_network.regularization.Regularization;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNVector;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class RecurrentLayer extends RecurrentNeuralLayer {
    protected NNVector[][] inputHidden;
    protected NNVector[][] outputHidden;

    private NNMatrix weightInput;
    private NNMatrix derWeightInput;

    private NNMatrix weightHidden;
    private NNMatrix derWeightHidden;

    private NNVector threshold;
    private NNVector derThreshold;

    private FunctionActivation functionActivation;

    public RecurrentLayer(int countNeuron) {
        this(countNeuron, 0);
    }

    public RecurrentLayer(RecurrentLayer layer) {
        this(layer.countNeuron, layer.recurrentDropout, layer.returnSequences);
        this.copy(layer);
    }

    public RecurrentLayer(int countNeuron, double recurrentDropout) {
        super(countNeuron, recurrentDropout);

        this.functionActivation = new FunctionActivation.Tanh();
    }

    public RecurrentLayer(int countNeuron, double recurrentDropout, boolean returnSequences) {
        this(countNeuron, recurrentDropout);
        setReturnSequences(returnSequences);
    }

    public RecurrentLayer setReturnSequences(boolean returnSequences) {
        this.returnSequences = returnSequences;

        return this;
    }

    @Override
    public void generateOutput(NNArray[] inputs, NNArray[][] state) {
        this.input = NNArrays.isMatrix(inputs);
        this.output = new NNMatrix[inputs.length];
        this.inputHidden = new NNVector[inputs.length][];
        this.outputHidden = new NNVector[inputs.length][];
        this.state = new NNVector[input.length][1];
        this.inputState = new NNVector[input.length][];

        ExecutorService executor = Executors.newFixedThreadPool(inputs.length);
        for (int cor = 0; cor < inputs.length; cor++) {
            final int i = cor;
            executor.execute(() -> {
                if (state != null) {
                    inputState[i] = NNArrays.isVector(state[i]);
                } else {
                    inputState[i] = null;
                }
                generateOutput(i, inputState[i]);
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    @Override
    public void generateError(NNArray[] errors, NNArray[][] errorsState) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNMatrix[input.length];
        this.errorState = new NNVector[input.length][1];

        ExecutorService executor = Executors.newFixedThreadPool(errors.length);
        for (int cor = 0; cor < errors.length; cor++) {
            final int i = cor;
            executor.execute(() -> {
                if (errorsState != null) {
                    generateError(i, NNArrays.isVector(errorsState[i]));
                } else {
                    generateError(i, null);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        //regularization derivative weightAttention
        if (trainable && regularization != null) {
            regularization.regularization(weightInput);
            regularization.regularization(weightHidden);
            regularization.regularization(threshold);
        }
    }

    public RecurrentLayer setFunctionActivation(FunctionActivation functionActivation) {
        this.functionActivation = functionActivation;

        return this;
    }

    @Override
    public void initialize(int[] size) {
        super.initialize(size);

        derThreshold = new NNVector(countNeuron);
        derWeightInput = new NNMatrix(countNeuron, depth);
        derWeightHidden = new NNMatrix(countNeuron, countNeuron);

        if (!loadWeight) {
            threshold = new NNVector(countNeuron);
            weightInput = new NNMatrix(countNeuron, depth);
            weightHidden = new NNMatrix(countNeuron, countNeuron);

            initializerInput.initialize(weightInput);
            initializerHidden.initialize(weightHidden);
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        optimizer.addDataOptimize(weightInput, derWeightInput);
        optimizer.addDataOptimize(weightHidden, derWeightHidden);
        optimizer.addDataOptimize(threshold, derThreshold);
    }

    @Override
    public int info() {
        int countParam = weightHidden.size() + weightInput.size() + threshold.size();
        System.out.println("Recurrent\t|  " + width + ",\t" + depth + "\t\t|  " + outWidth + ",\t" + countNeuron + "\t\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("Recurrent layer\n");
        writer.write(countNeuron + "\n");
        writer.write(recurrentDropout + "\n");
        writer.write(returnSequences + "\n");

        functionActivation.save(writer);

        threshold.save(writer);
        weightInput.save(writer);
        weightHidden.save(writer);

        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    private void generateOutput(int i, NNVector[] states) {
        int countRow = (returnSequences) ? input[i].getRow() : 1;
        output[i] = new NNMatrix(countRow, countNeuron);

        inputHidden[i] = new NNVector[input[i].getRow()];
        outputHidden[i] = new NNVector[input[i].getRow()];

        //pass through time
        for (int t = 0, tOut = 0; t < input[i].getRow(); t++) {
            inputHidden[i][t] = new NNVector(countNeuron);
            outputHidden[i][t] = new NNVector(countNeuron);

            NNVector hidden_t = null;
            if (t > 0) {
                hidden_t = outputHidden[i][t - 1];
            } else if (states != null) {
                hidden_t = states[0];
            }

            inputHidden[i][t].set(threshold);
            inputHidden[i][t].addMulRowToMatrix(input[i], t, weightInput);
            if (hidden_t != null) {
                inputHidden[i][t].addMul(hidden_t, weightHidden);
            }
            //activation hidden state
            functionActivation.activation(inputHidden[i][t], outputHidden[i][t]);
            //dropout hidden state
            if (dropout) {
                outputHidden[i][t].dropout(outputHidden[i][t], recurrentDropout);
            }
            //if return sequence pass current hidden state to output
            if (returnSequences || t == input[i].getRow() - 1) {
                output[i].set(outputHidden[i][t], tOut);
                tOut++;
            }
        }
        //if layer return state,than save last hidden state
        state[i][0] = outputHidden[i][input[i].getRow() - 1];
    }

    private void generateError(int i, NNVector[] errorState) {
        this.error[i] = new NNMatrix(input[i]);
        NNVector hiddenError = new NNVector(countNeuron);
        NNVector hiddenDelta = new NNVector(countNeuron);

        //copy error from next layer
        int tError = (returnSequences) ? outputHidden[i].length - 1 : 0;
        if (errorNL != null) {
            hiddenError.setRowFromMatrix(errorNL[i], tError);
        }
        if (errorState != null) {
            hiddenError.add(errorState[0]);
        }

        //pass through time
        for (int t = outputHidden[i].length - 1; t >= 0; t--) {
            //dropout back for error
            if (recurrentDropout != 0) {
                hiddenError.dropoutBack(outputHidden[i][t], hiddenError, recurrentDropout);
            }
            //derivative activation for current time step
            functionActivation.derivativeActivation(inputHidden[i][t], outputHidden[i][t], hiddenError, hiddenDelta);
            //find derivative for weightAttention
            if (trainable) {
                if (t > 0) {
                    derivativeWeight(input[i], t, outputHidden[i][t - 1], hiddenDelta);
                } else if (inputState[i] != null) {
                    derivativeWeight(input[i], t, inputState[i][0], hiddenDelta);
                }
            }
            //find error for previous time step
            //get error for current time step from next layer
            if (returnSequences && t > 0 && errorNL != null) {
                hiddenError.setRowFromMatrix(errorNL[i], t - 1);
            } else {
                hiddenError.clear();
            }
            hiddenError.addMulT(hiddenDelta, weightHidden);

            //find error for previous layer
            error[i].addMulT(t, hiddenDelta, weightInput);
            //add error to hidden state previous layer
        }
        this.errorState[i][0] = hiddenError;
    }

    private void derivativeWeight(NNMatrix input, int t, NNVector hidden_t, NNVector hiddenDelta) {
        derThreshold.add(hiddenDelta);
        int indexHWeight = 0, indexIWeight = 0, indexInput;

        for (int k = 0; k < hiddenDelta.size(); k++) {
            indexInput = input.getRowIndex()[t];
            if (hidden_t != null) {
                //find derivative for hidden weightAttention
                for (int m = 0; m < countNeuron; m++, indexHWeight++) {
                    derWeightHidden.getData()[indexHWeight] += hiddenDelta.get(k) * hidden_t.get(m);
                }
            }
            //find derivative for input's weightAttention
            for (int m = 0; m < input.getColumn(); m++, indexIWeight++, indexInput++) {
                derWeightInput.getData()[indexIWeight] += hiddenDelta.get(k) * input.getData()[indexInput];
            }
        }
    }

    public RecurrentLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;

        return this;
    }

    public RecurrentLayer setInitializer(Initializer initializer) {
        this.initializerInput = initializer;
        this.initializerHidden = initializer;

        return this;
    }

    public RecurrentLayer setInitializerHidden(Initializer initializer) {
        this.initializerHidden = initializer;

        return this;
    }

    public RecurrentLayer setInitializerInput(Initializer initializer) {
        this.initializerInput = initializer;

        return this;
    }

    public RecurrentLayer setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public static RecurrentLayer read(Scanner scanner) {
        RecurrentLayer recurrentLayer = new RecurrentLayer(Integer.parseInt(scanner.nextLine()),
                Double.parseDouble(scanner.nextLine()),
                Boolean.parseBoolean(scanner.nextLine()));

        recurrentLayer.functionActivation = FunctionActivation.read(scanner);

        recurrentLayer.threshold = NNVector.read(scanner);
        recurrentLayer.weightInput = NNMatrix.read(scanner);
        recurrentLayer.weightHidden = NNMatrix.read(scanner);

        recurrentLayer.setRegularization(Regularization.read(scanner));
        recurrentLayer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        recurrentLayer.loadWeight = true;
        return recurrentLayer;
    }
}