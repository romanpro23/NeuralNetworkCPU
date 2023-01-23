package neural_network.layers.recurrent;//package neural_network.layers.recurrent;
//
//import neural_network.activation.FunctionActivation;
//import neural_network.initialization.Initializer;
//import neural_network.optimizers.Optimizer;
//import neural_network.regularization.Regularization;
//import nnarrays.*;
//
//import java.io.FileWriter;
//import java.io.IOException;
//import java.util.Arrays;
//import java.util.Scanner;
//import java.util.concurrent.ExecutorService;
//import java.util.concurrent.Executors;
//
//public class ConvolutionLSTMLayer extends RecurrentConvolutionNeuralLayer {
//    private NNTensor[][] hiddenSMemory;
//
//    private NNTensor[][] gateFInput;
//    private NNTensor[][] gateFOutput;
//    private NNTensor[][] gateIInput;
//    private NNTensor[][] gateIOutput;
//    private NNTensor[][] gateOInput;
//    private NNTensor[][] gateOOutput;
//    private NNTensor[][] gateCInput;
//    private NNTensor[][] gateCOutput;
//
//    private NNTensor[] hiddenLongError;
//    private NNTensor[] hiddenLongDelta;
//    private NNTensor[] gateFDelta;
//    private NNTensor[] gateFError;
//    private NNTensor[] gateIDelta;
//    private NNTensor[] gateIError;
//    private NNTensor[] gateODelta;
//    private NNTensor[] gateOError;
//    private NNTensor[] gateCDelta;
//    private NNTensor[] gateCError;
//
//    private NNTensor4D[] weightInput;
//    private NNTensor4D[] derWeightInput;
//
//    private NNTensor4D[] weightHidden;
//    private NNTensor4D[] derWeightHidden;
//
//    private NNVector[] threshold;
//    private NNVector[] derThreshold;
//
//    private final FunctionActivation functionActivationSigmoid;
//    private final FunctionActivation functionActivationTanh;
//
//    //    public ConvolutionLSTMLayer(int countHead) {
////        this(countHead, 0);
////    }
////
////    public ConvolutionLSTMLayer(ConvolutionLSTMLayer layer) {
////        this(layer.countHead, layer.recurrentDropout, layer.returnSequences);
////        this.copy(layer);
////    }
////
//    public ConvolutionLSTMLayer(int countKernel, int heightKernel, int widthKernel, int step, int paddingY, int paddingX, double recurrentDropout) {
//        super(countKernel, heightKernel, widthKernel, step, paddingY, paddingX, recurrentDropout);
//
//        this.functionActivationTanh = new FunctionActivation.Tanh();
//        this.functionActivationSigmoid = new FunctionActivation.Sigmoid();
//    }
//
////    public ConvolutionLSTMLayer(int countHead, double recurrentDropout, boolean returnSequences) {
////        this(countHead, recurrentDropout);
////        setReturnSequences(returnSequences);
////    }
//
//    public ConvolutionLSTMLayer setReturnSequences(boolean returnSequences) {
//        this.returnSequences = returnSequences;
//
//        return this;
//    }
//
//    public ConvolutionLSTMLayer setPreLayer(RecurrentConvolutionNeuralLayer layer) {
//        super.setPreLayer(layer);
//
//        return this;
//    }
//
//    @Override
//    public void initialize(int[] size) {
//        super.initialize(size);
//
//        derThreshold = new NNVector[4];
//        derWeightInput = new NNTensor4D[4];
//        derWeightHidden = new NNTensor4D[4];
//
//        for (int i = 0; i < 4; i++) {
//            derThreshold[i] = new NNVector(countKernel);
//            derWeightInput[i] = new NNTensor4D(countKernel, heightKernel, widthKernel, depth);
//            derWeightHidden[i] = new NNTensor4D(countKernel, heightKernel, widthKernel, countKernel);
//        }
//
//        if (!loadWeight) {
//            threshold = new NNVector[4];
//            weightInput = new NNTensor4D[4];
//            weightHidden = new NNTensor4D[4];
//
//            for (int i = 0; i < 4; i++) {
//                threshold[i] = new NNVector(countKernel);
//                weightInput[i] = new NNTensor4D(countKernel, heightKernel, widthKernel, depth);
//                weightHidden[i] = new NNTensor4D(countKernel, heightKernel, widthKernel, countKernel);
//
//                initializerInput.initialize(weightInput[i]);
//                initializerHidden.initialize(weightHidden[i]);
//            }
//        }
//    }
//
//    @Override
//    public void initialize(Optimizer optimizer) {
//        for (int i = 0; i < 4; i++) {
//            optimizer.addDataOptimize(weightInput[i], derWeightInput[i]);
//            optimizer.addDataOptimize(weightHidden[i], derWeightHidden[i]);
//            optimizer.addDataOptimize(threshold[i], derThreshold[i]);
//        }
//    }
//
//    @Override
//    public int info() {
//        int countParam = (weightHidden[0].size() + weightInput[0].size() + threshold[0].size()) * 4;
//        System.out.println("ConvLSTM\t| " + height + ",\t" + width + ",\t" + depth + "\t| "
//                + outHeight + ",\t" + outWidth + ",\t" + outDepth + "\t|\t" + countParam);
//        return countParam;
//    }
//
//    @Override
//    public void write(FileWriter writer) throws IOException {
//        writer.write("Convolution LSTM layer\n");
//        writer.write(countKernel + " " + heightKernel + " " + widthKernel + " " + step + " " + paddingY + " " + paddingX + "\n");
//        writer.write(recurrentDropout + "\n");
//        writer.write(returnSequences + "\n");
//        writer.write(returnState + "\n");
//
//        for (int i = 0; i < 4; i++) {
//            threshold[i].save(writer);
//            weightInput[i].save(writer);
//            weightHidden[i].save(writer);
//        }
//
//        if (regularization != null) {
//            regularization.write(writer);
//        } else {
//            writer.write("null\n");
//        }
//        writer.write(trainable + "\n");
//        writer.flush();
//    }
//
//    @Override
//    public void generateOutput(NNArray[] inputs) {
//        this.input = NNArrays.isTensor4D(inputs);
//        this.output = new NNTensor4D[inputs.length];
//        this.inputHidden = new NNTensor[inputs.length][];
//        this.outputHidden = new NNTensor[inputs.length][];
//
//        this.hiddenSMemory = new NNTensor[inputs.length][];
//        this.gateIInput = new NNTensor[inputs.length][];
//        this.gateIOutput = new NNTensor[inputs.length][];
//        this.gateFInput = new NNTensor[inputs.length][];
//        this.gateFOutput = new NNTensor[inputs.length][];
//        this.gateOInput = new NNTensor[inputs.length][];
//        this.gateOOutput = new NNTensor[inputs.length][];
//        this.gateCInput = new NNTensor[inputs.length][];
//        this.gateCOutput = new NNTensor[inputs.length][];
//        if (returnState) {
//            this.state = new NNTensor[inputs.length][2];
//        }
//
//        int countC = getCountCores();
//        ExecutorService executor = Executors.newFixedThreadPool(countC);
//        for (int cor = 0; cor < countC; cor++) {
//            final int firstIndex = cor * input.length / countC;
//            final int lastIndex = Math.min(input.length, (cor + 1) * input.length / countC);
//            executor.execute(() -> {
//                for (int i = firstIndex; i < lastIndex; i++) {
//                    generateOutput(i);
//                }
//            });
//        }
//        executor.shutdown();
//        while (!executor.isTerminated()) {
//        }
//    }
//
//    private void generateOutput(int i) {
//        int countRow = (returnSequences) ? input[i].getRow() : 1;
//        output[i] = new NNTensor4D(countRow, outHeight, outWidth, outDepth);
//
//        inputHidden[i] = new NNTensor[input[i].getRow()];
//
//        this.hiddenSMemory[i] = new NNTensor[input[i].getRow()];
//        this.gateIInput[i] = new NNTensor[input[i].getRow()];
//        this.gateIOutput[i] = new NNTensor[input[i].getRow()];
//        this.gateFInput[i] = new NNTensor[input[i].getRow()];
//        this.gateFOutput[i] = new NNTensor[input[i].getRow()];
//        this.gateOInput[i] = new NNTensor[input[i].getRow()];
//        this.gateOOutput[i] = new NNTensor[input[i].getRow()];
//        this.gateCInput[i] = new NNTensor[input[i].getRow()];
//        this.gateCOutput[i] = new NNTensor[input[i].getRow()];
//
//        //pass through time
//        for (int t = 0, tOut = 0; t < input[i].getRow(); t++) {
//            inputHidden[i][t] = new  NNTensor(outHeight, outWidth, outDepth);
//            outputHidden[i][t] = new NNTensor(outHeight, outWidth, outDepth);
//
//            this.hiddenSMemory[i][t] = new NNTensor(outHeight, outWidth, outDepth);
//            this.gateIInput[i][t] = new NNTensor(outHeight, outWidth, outDepth);
//            this.gateIOutput[i][t] = new NNTensor(outHeight, outWidth, outDepth);
//            this.gateFInput[i][t] = new NNTensor(outHeight, outWidth, outDepth);
//            this.gateFOutput[i][t] = new NNTensor(outHeight, outWidth, outDepth);
//            this.gateOInput[i][t] = new NNTensor(outHeight, outWidth, outDepth);
//            this.gateOOutput[i][t] = new NNTensor(outHeight, outWidth, outDepth);
//            this.gateCInput[i][t] = new NNTensor(outHeight, outWidth, outDepth);
//            this.gateCOutput[i][t] = new NNTensor(outHeight, outWidth, outDepth);
//
//            NNTensor hiddenS_t = null;
//            NNTensor hiddenL_t = null;
//            if (t > 0) {
//                hiddenS_t = hiddenSMemory[i][t - 1];
//                hiddenL_t = inputHidden[i][t - 1];
//            } else if (hasPreLayer()) {
//                hiddenS_t = preLayer.state[i][0];
//                hiddenL_t = preLayer.state[i][1];
//            }
//
//            //generate new hiddenSMemory state for update and reset gate
//            gateFInput[i][t].set(threshold[0]);
//            gateIInput[i][t].set(threshold[1]);
//            gateOInput[i][t].set(threshold[2]);
//            gateCInput[i][t].set(threshold[3]);
//
//            gateFInput[i][t].addMulRowToMatrix(input[i], t, weightInput[0]);
//            gateIInput[i][t].addMulRowToMatrix(input[i], t, weightInput[1]);
//            gateOInput[i][t].addMulRowToMatrix(input[i], t, weightInput[2]);
//            gateCInput[i][t].addMulRowToMatrix(input[i], t, weightInput[3]);
//            if (hiddenS_t != null) {
//                gateFInput[i][t].addMul(hiddenS_t, weightHidden[0]);
//                gateIInput[i][t].addMul(hiddenS_t, weightHidden[1]);
//                gateOInput[i][t].addMul(hiddenS_t, weightHidden[2]);
//                gateCInput[i][t].addMul(hiddenS_t, weightHidden[3]);
//            }
//
//            //activation gate
//            functionActivationSigmoid.activation(gateFInput[i][t], gateFOutput[i][t]);
//            functionActivationSigmoid.activation(gateIInput[i][t], gateIOutput[i][t]);
//            functionActivationSigmoid.activation(gateOInput[i][t], gateOOutput[i][t]);
//            functionActivationTanh.activation(gateCInput[i][t], gateCOutput[i][t]);
//
//            // find current long memory
//            inputHidden[i][t].mulVectors(gateIOutput[i][t], gateCOutput[i][t]);
//            if (hiddenL_t != null) {
//                inputHidden[i][t].addProduct(hiddenL_t, gateFOutput[i][t]);
//            }
//            functionActivationTanh.activation(inputHidden[i][t], outputHidden[i][t]);
//
//            hiddenSMemory[i][t].mulVectors(gateOOutput[i][t], outputHidden[i][t]);
//
//            //dropout hiddenSMemory state
//            if (dropout) {
//                hiddenSMemory[i][t].dropout(hiddenSMemory[i][t], recurrentDropout);
//            }
//            //if return sequence pass current hiddenSMemory state to output
//            if (returnSequences || t == input[i].getRow() - 1) {
//                output[i].set(hiddenSMemory[i][t], tOut);
//                tOut++;
//            }
//        }
//        //if layer return state,than save last hiddenSMemory state
//        if (returnState) {
//            state[i][0] = hiddenSMemory[i][input[i].getRow() - 1];
//            state[i][1] = inputHidden[i][input[i].getRow() - 1];
//        }
//    }
//
//    @Override
//    public void generateError(NNArray[] errors) {
//        errorNL = getErrorNextLayer(errors);
//        this.error = new NNMatrix[input.length];
//        this.hiddenError = new NNVector[input.length];
//        if (hasPreLayer()) {
//            this.errorState = new NNVector[input.length][2];
//        }
//
//        gateFDelta = new NNVector[input.length];
//        gateFError = new NNVector[input.length];
//        gateIDelta = new NNVector[input.length];
//        gateIError = new NNVector[input.length];
//        gateODelta = new NNVector[input.length];
//        gateOError = new NNVector[input.length];
//        gateCDelta = new NNVector[input.length];
//        gateCError = new NNVector[input.length];
//        gateCError = new NNVector[input.length];
//        hiddenLongDelta = new NNVector[input.length];
//        hiddenLongError = new NNVector[input.length];
//
//        int countC = getCountCores();
//        ExecutorService executor = Executors.newFixedThreadPool(countC);
//        for (int cor = 0; cor < countC; cor++) {
//            final int firstIndex = cor * input.length / countC;
//            final int lastIndex = Math.min(input.length, (cor + 1) * input.length / countC);
//            executor.execute(() -> {
//                for (int i = firstIndex; i < lastIndex; i++) {
//                    generateError(i);
//                }
//            });
//        }
//        executor.shutdown();
//        while (!executor.isTerminated()) {
//        }
//
//        //regularization derivative weightAttention
//        if (regularization != null) {
//            for (int i = 0; i < 4; i++) {
//                regularization.regularization(weightInput[i]);
//                regularization.regularization(weightHidden[i]);
//                regularization.regularization(threshold[i]);
//            }
//        }
//    }
//
//    private void generateError(int i) {
//        this.error[i] = new NNMatrix(input[i]);
//        hiddenError[i] = new NNVector(countHead);
//        hiddenLongDelta[i] = new NNVector(countHead);
//        hiddenLongError[i] = new NNVector(countHead);
//
//        gateFDelta[i] = new NNVector(countHead);
//        gateFError[i] = new NNVector(countHead);
//        gateIDelta[i] = new NNVector(countHead);
//        gateIError[i] = new NNVector(countHead);
//        gateODelta[i] = new NNVector(countHead);
//        gateOError[i] = new NNVector(countHead);
//        gateCDelta[i] = new NNVector(countHead);
//        gateCError[i] = new NNVector(countHead);
//
//        //copy error from next layer
//        int tError = (returnSequences) ? hiddenSMemory[i].length - 1 : 0;
//        if (errorNL != null) {
//            hiddenError[i].setRowFromMatrix(errorNL[i], tError);
//        }
//        if (returnState) {
//            hiddenError[i].add(nextLayer.errorState[i][0]);
//            hiddenLongError[i].set(nextLayer.errorState[i][1]);
//        }
//
//        //pass through time
//        for (int t = input[i].getRow() - 1; t >= 0; t--) {
//            NNVector hiddenS_t = null;
//            NNVector hiddenL_t = null;
//            if (t > 0) {
//                hiddenS_t = hiddenSMemory[i][t - 1];
//                hiddenL_t = inputHidden[i][t - 1];
//            } else if (hasPreLayer()) {
//                hiddenS_t = preLayer.state[i][0];
//                hiddenL_t = preLayer.state[i][1];
//            }
//            //dropout back for error
//            hiddenError[i].dropoutBack(hiddenSMemory[i][t], hiddenError[i], recurrentDropout);
//            //find error for long memory
//            functionActivationTanh.derivativeActivation(inputHidden[i][t], outputHidden[i][t], hiddenError[i], hiddenLongDelta[i]);
//            hiddenLongDelta[i].mul(gateOOutput[i][t]);
//            hiddenLongDelta[i].add(hiddenLongError[i]);
//
//            gateOError[i].mulVectors(hiddenError[i], outputHidden[i][t]);
//            gateCError[i].mulVectors(hiddenLongDelta[i], gateIOutput[i][t]);
//            gateIError[i].mulVectors(hiddenLongDelta[i], gateCOutput[i][t]);
//            gateFDelta[i].clear();
//            if (hiddenL_t != null) {
//                gateFError[i].mulVectors(hiddenLongDelta[i], hiddenL_t);
//                functionActivationSigmoid.derivativeActivation(gateFInput[i][t], gateFOutput[i][t], gateFError[i], gateFDelta[i]);
//            }
//
//            functionActivationSigmoid.derivativeActivation(gateIInput[i][t], gateIOutput[i][t], gateIError[i], gateIDelta[i]);
//            functionActivationSigmoid.derivativeActivation(gateOInput[i][t], gateOOutput[i][t], gateOError[i], gateODelta[i]);
//            functionActivationTanh.derivativeActivation(gateCInput[i][t], gateCOutput[i][t], gateCError[i], gateCDelta[i]);
//
//            //find derivative for weightAttention
//            if (trainable) {
//                derivativeWeight(t, i, hiddenS_t);
//            }
//
//            //find error for previous time step
//            hiddenLongError[i].mulVectors(hiddenLongDelta[i], gateFOutput[i][t]);
//            if (returnSequences && t > 0 && errorNL != null) {
//                hiddenError[i].setRowFromMatrix(errorNL[i], t - 1);
//            } else {
//                hiddenError[i].clear();
//            }
//            hiddenError[i].addMulT(gateFDelta[i], weightHidden[0]);
//            hiddenError[i].addMulT(gateIDelta[i], weightHidden[1]);
//            hiddenError[i].addMulT(gateODelta[i], weightHidden[2]);
//            hiddenError[i].addMulT(gateCDelta[i], weightHidden[3]);
//
//            //find error for previous layer
//            error[i].addMulT(t, gateFDelta[i], weightInput[0]);
//            error[i].addMulT(t, gateIDelta[i], weightInput[1]);
//            error[i].addMulT(t, gateODelta[i], weightInput[2]);
//            error[i].addMulT(t, gateCDelta[i], weightInput[3]);
//        }
//        if (hasPreLayer()) {
//            errorState[i][0] = new NNVector(countHead);
//            errorState[i][1] = new NNVector(countHead);
//            errorState[i][0].set(this.hiddenError[i]);
//            errorState[i][1].set(this.hiddenLongError[i]);
//        }
//    }
//
//    private void derivativeWeight(int t, int i, NNVector hidden_t) {
//        derThreshold[0].add(gateFDelta[i]);
//        derThreshold[1].add(gateIDelta[i]);
//        derThreshold[2].add(gateODelta[i]);
//        derThreshold[3].add(gateCDelta[i]);
//        int indexHWeight = 0, indexIWeight = 0, indexInput;
//
//        for (int k = 0; k < hiddenSMemory[i][t].size(); k++) {
//            indexInput = input[i].getRowIndex()[t];
//
//            if (hidden_t != null) {
//                //find derivative for hiddenSMemory weightAttention
//                for (int m = 0; m < countHead; m++, indexHWeight++) {
//                    derWeightHidden[0].getData()[indexHWeight] += gateFDelta[i].get(k) * hidden_t.get(m);
//                    derWeightHidden[1].getData()[indexHWeight] += gateIDelta[i].get(k) * hidden_t.get(m);
//                    derWeightHidden[2].getData()[indexHWeight] += gateODelta[i].get(k) * hidden_t.get(m);
//                    derWeightHidden[3].getData()[indexHWeight] += gateCDelta[i].get(k) * hidden_t.get(m);
//                }
//            }
//            //find derivative for input's weightAttention
//            for (int m = 0; m < input[i].getColumn(); m++, indexIWeight++, indexInput++) {
//                derWeightInput[0].getData()[indexIWeight] += gateFDelta[i].get(k) * input[i].getData()[indexInput];
//                derWeightInput[1].getData()[indexIWeight] += gateIDelta[i].get(k) * input[i].getData()[indexInput];
//                derWeightInput[2].getData()[indexIWeight] += gateODelta[i].get(k) * input[i].getData()[indexInput];
//                derWeightInput[3].getData()[indexIWeight] += gateCDelta[i].get(k) * input[i].getData()[indexInput];
//            }
//        }
//    }
//
//    public ConvolutionLSTMLayer setRegularization(Regularization regularization) {
//        this.regularization = regularization;
//
//        return this;
//    }
//
//    public ConvolutionLSTMLayer setInitializer(Initializer initializer) {
//        this.initializerInput = initializer;
//        this.initializerHidden = initializer;
//
//        return this;
//    }
//
//    public ConvolutionLSTMLayer setInitializerInput(Initializer initializer) {
//        this.initializerInput = initializer;
//
//        return this;
//    }
//
//    public ConvolutionLSTMLayer setInitializerHidden(Initializer initializer) {
//        this.initializerHidden = initializer;
//
//        return this;
//    }
//
//    public ConvolutionLSTMLayer setTrainable(boolean trainable) {
//        this.trainable = trainable;
//
//        return this;
//    }
//
//    public static ConvolutionLSTMLayer read(Scanner scanner) {
//        int[] param = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
//        ConvolutionLSTMLayer recurrentLayer = new ConvolutionLSTMLayer(param[0], param[1], param[2], param[3], param[4], param[5],
//                Double.parseDouble(scanner.nextLine()));
//
//        recurrentLayer.returnSequences = Boolean.parseBoolean(scanner.nextLine());
//        recurrentLayer.returnState = Boolean.parseBoolean(scanner.nextLine());
//
//        recurrentLayer.threshold = new NNVector[4];
//        recurrentLayer.weightInput = new NNTensor4D[4];
//        recurrentLayer.weightHidden = new NNTensor4D[4];
//
//        for (int i = 0; i < 4; i++) {
//            recurrentLayer.threshold[i] = NNVector.read(scanner);
//            recurrentLayer.weightInput[i] = NNTensor4D.read(scanner);
//            recurrentLayer.weightHidden[i] = NNTensor4D.read(scanner);
//        }
//
//        recurrentLayer.setRegularization(Regularization.read(scanner));
//        recurrentLayer.setTrainable(Boolean.parseBoolean(scanner.nextLine()));
//        recurrentLayer.loadWeight = true;
//        return recurrentLayer;
//    }
//}