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

public class LSTMLuongAttentionLayer extends LuongAttentionLayer {
    private NNVector[][] hiddenSMemory;

    private NNVector[][] gateFInput;
    private NNVector[][] gateFOutput;
    private NNVector[][] gateIInput;
    private NNVector[][] gateIOutput;
    private NNVector[][] gateOInput;
    private NNVector[][] gateOOutput;
    private NNVector[][] gateCInput;
    private NNVector[][] gateCOutput;

    private NNVector[] hiddenLongError;
    private NNVector[] hiddenLongDelta;
    private NNVector[] gateFDelta;
    private NNVector[] gateFError;
    private NNVector[] gateIDelta;
    private NNVector[] gateIError;
    private NNVector[] gateODelta;
    private NNVector[] gateOError;
    private NNVector[] gateCDelta;
    private NNVector[] gateCError;

    private NNMatrix[] weightInput;
    private NNMatrix[] derWeightInput;

    private NNMatrix[] weightHidden;
    private NNMatrix[] derWeightHidden;

    private NNVector[] threshold;
    private NNVector[] derThreshold;

    private final FunctionActivation functionActivationSigmoid;
    private final FunctionActivation functionActivationTanh;

    public LSTMLuongAttentionLayer(int countNeuron) {
        this(countNeuron, 0, Attention.DOT);
    }

    public LSTMLuongAttentionLayer(LSTMLuongAttentionLayer layer) {
        this(layer.countNeuron, layer.recurrentDropout, layer.attention, layer.returnSequences);
        this.copy(layer);
    }

    public LSTMLuongAttentionLayer(int countNeuron, double recurrentDropout, LuongAttentionLayer.Attention attention) {
        super(countNeuron, recurrentDropout, attention);

        this.functionActivationTanh = new FunctionActivation.Tanh();
        this.functionActivationSigmoid = new FunctionActivation.Sigmoid();
    }

    public LSTMLuongAttentionLayer(int countNeuron, double recurrentDropout, LuongAttentionLayer.Attention attention, boolean returnSequences) {
        this(countNeuron, recurrentDropout, attention);
        setReturnSequences(returnSequences);
    }

    public LSTMLuongAttentionLayer setReturnSequences(boolean returnSequences) {
        this.returnSequences = returnSequences;

        return this;
    }

    public LSTMLuongAttentionLayer setPreLayer(RecurrentNeuralLayer layer) {
        super.setPreLayer(layer);
        layer.addNextLayer(this);

        return this;
    }

    @Override
    public void initialize(int[] size) {
        super.initialize(size);

        derThreshold = new NNVector[4];
        derWeightInput = new NNMatrix[4];
        derWeightHidden = new NNMatrix[4];

        for (int i = 0; i < 4; i++) {
            derThreshold[i] = new NNVector(countNeuron);
            derWeightInput[i] = new NNMatrix(countNeuron, depth);
            derWeightHidden[i] = new NNMatrix(countNeuron, countNeuron);
        }

        if (!loadWeight) {
            threshold = new NNVector[4];
            weightInput = new NNMatrix[4];
            weightHidden = new NNMatrix[4];

            for (int i = 0; i < 4; i++) {
                threshold[i] = new NNVector(countNeuron);
                weightInput[i] = new NNMatrix(countNeuron, depth);
                weightHidden[i] = new NNMatrix(countNeuron, countNeuron);

                initializerInput.initialize(weightInput[i]);
                initializerHidden.initialize(weightHidden[i]);
            }

            if (attention == Attention.GENERAL) {
                initializerInput.initialize(weightVal);
            } else if (attention == Attention.CONCAT) {
                initializerInput.initialize(weightAttention);
                initializerInput.initialize(weightVal);
            }
        }
    }

    @Override
    public void initialize(Optimizer optimizer) {
        super.initialize(optimizer);
        for (int i = 0; i < 4; i++) {
            optimizer.addDataOptimize(weightInput[i], derWeightInput[i]);
            optimizer.addDataOptimize(weightHidden[i], derWeightHidden[i]);
            optimizer.addDataOptimize(threshold[i], derThreshold[i]);
        }
    }

    @Override
    public int info() {
        int countParam = (weightHidden[0].size() + weightInput[0].size() + threshold[0].size()) * 4;
        if (attention == Attention.GENERAL) {
            countParam += weightVal.size();
        } else if (attention == Attention.CONCAT) {
            countParam += weightVal.size();
            countParam += weightAttention.size();
        }

        System.out.println("Luong LSTM\t|  " + width + ",\t" + depth + "\t\t|  " + outWidth + ",\t" + countNeuron + "\t\t|\t" + countParam);
        return countParam;
    }

    @Override
    public void save(FileWriter writer) throws IOException {
        writer.write("LSTM luong attention layer\n");
        writer.write(countNeuron + "\n");
        writer.write(recurrentDropout + "\n");
        writer.write(returnSequences + "\n");
        writer.write(returnState + "\n");

        for (int i = 0; i < 4; i++) {
            threshold[i].save(writer);
            weightInput[i].save(writer);
            weightHidden[i].save(writer);
        }

        if (attention == Attention.GENERAL) {
            weightVal.save(writer);
        } else if (attention == Attention.CONCAT) {
            weightVal.save(writer);
            weightAttention.save(writer);
        }

        if (regularization != null) {
            regularization.write(writer);
        } else {
            writer.write("null\n");
        }
        writer.write(trainable + "\n");
        writer.flush();
    }

    @Override
    public void generateOutput(NNArray[] inputs) {
        this.input = NNArrays.isMatrix(inputs);
        this.output = new NNMatrix[inputs.length];
        this.inputHidden = new NNVector[inputs.length][];
        this.outputHidden = new NNVector[inputs.length][];

        this.hiddenSMemory = new NNVector[inputs.length][];
        this.gateIInput = new NNVector[inputs.length][];
        this.gateIOutput = new NNVector[inputs.length][];
        this.gateFInput = new NNVector[inputs.length][];
        this.gateFOutput = new NNVector[inputs.length][];
        this.gateOInput = new NNVector[inputs.length][];
        this.gateOOutput = new NNVector[inputs.length][];
        this.gateCInput = new NNVector[inputs.length][];
        this.gateCOutput = new NNVector[inputs.length][];
        if (returnState) {
            this.state = new NNVector[inputs.length][2];
        }

        outputPreLayer = NNArrays.isMatrix(preLayer.getOutput());
        initializeMemory(inputs.length);

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int cor = 0; cor < countC; cor++) {
            final int firstIndex = cor * input.length / countC;
            final int lastIndex = Math.min(input.length, (cor + 1) * input.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    generateOutput(i);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
    }

    private void generateOutput(int i) {
        int countRow = (returnSequences) ? input[i].getRow() : 1;
        output[i] = new NNMatrix(countRow, countNeuron * 2);

        inputHidden[i] = new NNVector[input[i].getRow()];
        outputHidden[i] = new NNVector[input[i].getRow()];

        this.hiddenSMemory[i] = new NNVector[input[i].getRow()];
        this.gateIInput[i] = new NNVector[input[i].getRow()];
        this.gateIOutput[i] = new NNVector[input[i].getRow()];
        this.gateFInput[i] = new NNVector[input[i].getRow()];
        this.gateFOutput[i] = new NNVector[input[i].getRow()];
        this.gateOInput[i] = new NNVector[input[i].getRow()];
        this.gateOOutput[i] = new NNVector[input[i].getRow()];
        this.gateCInput[i] = new NNVector[input[i].getRow()];
        this.gateCOutput[i] = new NNVector[input[i].getRow()];

        initializeMemory(i, input[i].getRow());

        //pass through time
        for (int t = 0, tOut = 0; t < input[i].getRow(); t++) {
            inputHidden[i][t] = new NNVector(countNeuron);
            outputHidden[i][t] = new NNVector(countNeuron);

            this.hiddenSMemory[i][t] = new NNVector(countNeuron);
            this.gateIInput[i][t] = new NNVector(countNeuron);
            this.gateIOutput[i][t] = new NNVector(countNeuron);
            this.gateFInput[i][t] = new NNVector(countNeuron);
            this.gateFOutput[i][t] = new NNVector(countNeuron);
            this.gateOInput[i][t] = new NNVector(countNeuron);
            this.gateOOutput[i][t] = new NNVector(countNeuron);
            this.gateCInput[i][t] = new NNVector(countNeuron);
            this.gateCOutput[i][t] = new NNVector(countNeuron);

            NNVector hiddenS_t = null;
            NNVector hiddenL_t = null;
            if (t > 0) {
                hiddenS_t = hiddenSMemory[i][t - 1];
                hiddenL_t = inputHidden[i][t - 1];
            } else if (hasPreLayer()) {
                hiddenS_t = getStatePreLayer(i)[0];
                hiddenL_t = getStatePreLayer(i)[1];
            }

            //generate new hiddenSMemory state for update and reset gate
            gateFInput[i][t].set(threshold[0]);
            gateIInput[i][t].set(threshold[1]);
            gateOInput[i][t].set(threshold[2]);
            gateCInput[i][t].set(threshold[3]);

            gateFInput[i][t].addMulRowToMatrix(input[i], t, weightInput[0]);
            gateIInput[i][t].addMulRowToMatrix(input[i], t, weightInput[1]);
            gateOInput[i][t].addMulRowToMatrix(input[i], t, weightInput[2]);
            gateCInput[i][t].addMulRowToMatrix(input[i], t, weightInput[3]);
            if (hiddenS_t != null) {
                gateFInput[i][t].addMul(hiddenS_t, weightHidden[0]);
                gateIInput[i][t].addMul(hiddenS_t, weightHidden[1]);
                gateOInput[i][t].addMul(hiddenS_t, weightHidden[2]);
                gateCInput[i][t].addMul(hiddenS_t, weightHidden[3]);
            }

            //activation gate
            functionActivationSigmoid.activation(gateFInput[i][t], gateFOutput[i][t]);
            functionActivationSigmoid.activation(gateIInput[i][t], gateIOutput[i][t]);
            functionActivationSigmoid.activation(gateOInput[i][t], gateOOutput[i][t]);
            functionActivationTanh.activation(gateCInput[i][t], gateCOutput[i][t]);

            // find current long memory
            inputHidden[i][t].mulVectors(gateIOutput[i][t], gateCOutput[i][t]);
            if(hiddenL_t != null){
                inputHidden[i][t].addProduct(hiddenL_t, gateFOutput[i][t]);
            }
            functionActivationTanh.activation(inputHidden[i][t], outputHidden[i][t]);

            hiddenSMemory[i][t].mulVectors(gateOOutput[i][t], outputHidden[i][t]);

            //dropout hiddenSMemory state
            if (dropout) {
                hiddenSMemory[i][t].dropout(hiddenSMemory[i][t], recurrentDropout);
            }
            outputVector[i][t] = generateAttention(t, i, hiddenSMemory[i][t]).concat(hiddenSMemory[i][t]);
            //if return sequence pass current hiddenSMemory state to output
            if (returnSequences || t == input[i].getRow() - 1) {
                output[i].set(outputVector[i][t], tOut);
                tOut++;
            }
        }
        //if layer return state,than save last hiddenSMemory state
        if (returnState) {
            state[i][0] = hiddenSMemory[i][input[i].getRow() - 1];
            state[i][1] = inputHidden[i][input[i].getRow() - 1];
        }
    }

    @Override
    public void generateError(NNArray[] errors) {
        errorNL = getErrorNextLayer(errors);
        this.error = new NNMatrix[input.length];
        this.hiddenError = new NNVector[input.length];
        if (hasPreLayer()) {
            this.errorState = new NNVector[input.length][2];
        }

        gateFDelta = new NNVector[input.length];
        gateFError = new NNVector[input.length];
        gateIDelta = new NNVector[input.length];
        gateIError = new NNVector[input.length];
        gateODelta = new NNVector[input.length];
        gateOError = new NNVector[input.length];
        gateCDelta = new NNVector[input.length];
        gateCError = new NNVector[input.length];
        gateCError = new NNVector[input.length];
        hiddenLongDelta = new NNVector[input.length];
        hiddenLongError = new NNVector[input.length];

        int countC = getCountCores();
        ExecutorService executor = Executors.newFixedThreadPool(countC);
        for (int cor = 0; cor < countC; cor++) {
            final int firstIndex = cor * input.length / countC;
            final int lastIndex = Math.min(input.length, (cor + 1) * input.length / countC);
            executor.execute(() -> {
                for (int i = firstIndex; i < lastIndex; i++) {
                    generateError(i);
                }
            });
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        //regularization derivative weightAttention
        if (trainable && regularization != null) {
            for (int i = 0; i < 4; i++) {
                regularization.regularization(weightInput[i]);
                regularization.regularization(weightHidden[i]);
                regularization.regularization(threshold[i]);
            }
            if (attention == Attention.GENERAL) {
                regularization.regularization(weightVal);
            } else if (attention == Attention.CONCAT) {
                regularization.regularization(weightAttention);
                regularization.regularization(weightVal);
            }
        }
    }

    private void generateError(int i) {
        this.error[i] = new NNMatrix(input[i]);
        this.errorInput[i] = new NNMatrix(outputPreLayer[i]);
        hiddenError[i] = new NNVector(countNeuron);
        hiddenLongDelta[i] = new NNVector(countNeuron);
        hiddenLongError[i] = new NNVector(countNeuron);

        gateFDelta[i] = new NNVector(countNeuron);
        gateFError[i] = new NNVector(countNeuron);
        gateIDelta[i] = new NNVector(countNeuron);
        gateIError[i] = new NNVector(countNeuron);
        gateODelta[i] = new NNVector(countNeuron);
        gateOError[i] = new NNVector(countNeuron);
        gateCDelta[i] = new NNVector(countNeuron);
        gateCError[i] = new NNVector(countNeuron);

        //copy error from next layer
        int tError = (returnSequences) ? hiddenSMemory[i].length - 1 : 0;

        NNVector[] hiddenErrors = errorNL[i].toVectors();
        hiddenError[i] = hiddenErrors[tError].subVector(countNeuron, countNeuron);
        if (returnState) {
            hiddenError[i].add(getErrorStateNextLayer(i)[0]);
            hiddenLongError[i].set(getErrorStateNextLayer(i)[1]);
        }
        hiddenError[i].add(generateErrorAttention(hiddenErrors[tError].subVector(0, countNeuron), i, tError));

        //pass through time
        for (int t = input[i].getRow() - 1; t >= 0; t--) {
            NNVector hiddenS_t = null;
            NNVector hiddenL_t = null;
            if (t > 0) {
                hiddenS_t = hiddenSMemory[i][t - 1];
                hiddenL_t = inputHidden[i][t - 1];
            } else if (hasPreLayer()) {
                hiddenS_t = getStatePreLayer(i)[0];
                hiddenL_t = getStatePreLayer(i)[1];
            }
            //dropout back for error
            hiddenError[i].dropoutBack(hiddenSMemory[i][t], hiddenError[i], recurrentDropout);
            //find error for long memory
            functionActivationTanh.derivativeActivation(inputHidden[i][t], outputHidden[i][t], hiddenError[i], hiddenLongDelta[i]);
            hiddenLongDelta[i].mul(gateOOutput[i][t]);
            hiddenLongDelta[i].add(hiddenLongError[i]);

            gateOError[i].mulVectors(hiddenError[i], outputHidden[i][t]);
            gateCError[i].mulVectors(hiddenLongDelta[i], gateIOutput[i][t]);
            gateIError[i].mulVectors(hiddenLongDelta[i], gateCOutput[i][t]);
            gateFDelta[i].clear();
            if(hiddenL_t != null){
                gateFError[i].mulVectors(hiddenLongDelta[i], hiddenL_t);
                functionActivationSigmoid.derivativeActivation(gateFInput[i][t], gateFOutput[i][t], gateFError[i], gateFDelta[i]);
            }

            functionActivationSigmoid.derivativeActivation(gateIInput[i][t], gateIOutput[i][t], gateIError[i], gateIDelta[i]);
            functionActivationSigmoid.derivativeActivation(gateOInput[i][t], gateOOutput[i][t], gateOError[i], gateODelta[i]);
            functionActivationTanh.derivativeActivation(gateCInput[i][t], gateCOutput[i][t], gateCError[i], gateCDelta[i]);

            //find derivative for weightAttention
            if (trainable) {
                derivativeWeight(t, i, hiddenS_t);
            }

            //find error for previous time step
            hiddenLongError[i].mulVectors(hiddenLongDelta[i], gateFOutput[i][t]);
            if (returnSequences && t > 0 && errorNL != null) {
                hiddenError[i].add(hiddenErrors[t - 1].subVector(countNeuron, countNeuron));
                hiddenError[i].add(generateErrorAttention(hiddenErrors[t - 1].subVector(0, countNeuron), i, t - 1));
            } else {
                hiddenError[i].clear();
            }
            hiddenError[i].addMulT(gateFDelta[i], weightHidden[0]);
            hiddenError[i].addMulT(gateIDelta[i], weightHidden[1]);
            hiddenError[i].addMulT(gateODelta[i], weightHidden[2]);
            hiddenError[i].addMulT(gateCDelta[i], weightHidden[3]);

            //find error for previous layer
            error[i].addMulT(t, gateFDelta[i], weightInput[0]);
            error[i].addMulT(t, gateIDelta[i], weightInput[1]);
            error[i].addMulT(t, gateODelta[i], weightInput[2]);
            error[i].addMulT(t, gateCDelta[i], weightInput[3]);
        }
        if (hasPreLayer()) {
            errorState[i][0] = new NNVector(countNeuron);
            errorState[i][1] = new NNVector(countNeuron);
            errorState[i][0].set(this.hiddenError[i]);
            errorState[i][1].set(this.hiddenLongError[i]);
        }
    }

    private void derivativeWeight(int t, int i, NNVector hidden_t) {
        derThreshold[0].add(gateFDelta[i]);
        derThreshold[1].add(gateIDelta[i]);
        derThreshold[2].add(gateODelta[i]);
        derThreshold[3].add(gateCDelta[i]);
        int indexHWeight = 0, indexIWeight = 0, indexInput;

        for (int k = 0; k < hiddenSMemory[i][t].size(); k++) {
            indexInput = input[i].getRowIndex()[t];

            if (hidden_t != null) {
                //find derivative for hiddenSMemory weightAttention
                for (int m = 0; m < countNeuron; m++, indexHWeight++) {
                    derWeightHidden[0].getData()[indexHWeight] += gateFDelta[i].get(k) * hidden_t.get(m);
                    derWeightHidden[1].getData()[indexHWeight] += gateIDelta[i].get(k) * hidden_t.get(m);
                    derWeightHidden[2].getData()[indexHWeight] += gateODelta[i].get(k) * hidden_t.get(m);
                    derWeightHidden[3].getData()[indexHWeight] += gateCDelta[i].get(k) * hidden_t.get(m);
                }
            }
            //find derivative for input's weightAttention
            for (int m = 0; m < input[i].getColumn(); m++, indexIWeight++, indexInput++) {
                derWeightInput[0].getData()[indexIWeight] += gateFDelta[i].get(k) * input[i].getData()[indexInput];
                derWeightInput[1].getData()[indexIWeight] += gateIDelta[i].get(k) * input[i].getData()[indexInput];
                derWeightInput[2].getData()[indexIWeight] += gateODelta[i].get(k) * input[i].getData()[indexInput];
                derWeightInput[3].getData()[indexIWeight] += gateCDelta[i].get(k) * input[i].getData()[indexInput];
            }
        }
    }

    public LSTMLuongAttentionLayer setRegularization(Regularization regularization) {
        this.regularization = regularization;

        return this;
    }

    public LSTMLuongAttentionLayer setInitializer(Initializer initializer) {
        this.initializerInput = initializer;
        this.initializerHidden = initializer;

        return this;
    }

    public LSTMLuongAttentionLayer setInitializerInput(Initializer initializer) {
        this.initializerInput = initializer;

        return this;
    }

    public LSTMLuongAttentionLayer setInitializerHidden(Initializer initializer) {
        this.initializerHidden = initializer;

        return this;
    }

    public LSTMLuongAttentionLayer setTrainable(boolean trainable) {
        this.trainable = trainable;

        return this;
    }

    public static LSTMLuongAttentionLayer read(Scanner scanner) {
        LSTMLuongAttentionLayer recurrentLayer = new LSTMLuongAttentionLayer(Integer.parseInt(scanner.nextLine()),
                Double.parseDouble(scanner.nextLine()),
                readAttention(scanner.nextLine()),
                Boolean.parseBoolean(scanner.nextLine()));

        recurrentLayer.returnState = Boolean.parseBoolean(scanner.nextLine());

        recurrentLayer.threshold = new NNVector[4];
        recurrentLayer.weightInput = new NNMatrix[4];
        recurrentLayer.weightHidden = new NNMatrix[4];

        for (int i = 0; i < 4; i++) {
            recurrentLayer.threshold[i] = NNVector.read(scanner);
            recurrentLayer.weightInput[i] = NNMatrix.read(scanner);
            recurrentLayer.weightHidden[i] = NNMatrix.read(scanner);
        }

        if (recurrentLayer.attention == Attention.GENERAL) {
            recurrentLayer.weightVal = NNMatrix.read(scanner);
        } else if (recurrentLayer.attention == Attention.CONCAT) {
            recurrentLayer.weightVal = NNMatrix.read(scanner);
            recurrentLayer.weightAttention = NNMatrix.read(scanner);
        }

        recurrentLayer.setRegularization(Regularization.read(scanner));
        recurrentLayer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
        recurrentLayer.loadWeight = true;
        return recurrentLayer;
    }
}