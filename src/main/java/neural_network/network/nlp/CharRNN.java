package neural_network.network.nlp;

import data.network_train.NNData1D;
import neural_network.layers.recurrent.RecurrentNeuralLayer;
import neural_network.layers.reshape.EmbeddingLayer;
import neural_network.network.NeuralNetwork;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNMatrix;
import nnarrays.NNVector;

public class CharRNN {
    protected NeuralNetwork network;

    protected int sizeVoc;

    public CharRNN(NeuralNetwork network) {
        this.network = network;
        this.sizeVoc = ((EmbeddingLayer) network.getLayers().get(0)).getSizeVocabulary();
    }

    public float train(NNArray[] input) {
        NNData1D data = getInputVector(NNArrays.isVector(input));

        return network.train(data.getInput(), NNArrays.toHotVector(data.getOutput(), sizeVoc));
    }

    private NNData1D getInputVector(NNVector[] rawData) {
        NNVector[] input = new NNVector[rawData.length];
        NNVector[] output = new NNVector[rawData.length];
        for (int i = 0; i < rawData.length; i++) {
            if (rawData[i].size() > 1) {
                input[i] = rawData[i].subVector(0, rawData[i].size() - 1);
                output[i] = rawData[i].subVector(1, rawData[i].size() - 1);
            } else {
                input[i] = rawData[i];
                output[i] = rawData[i];
            }
        }

        return new NNData1D(input, output);
    }

    public NNVector query(NNVector input, int countChars) {
        ((RecurrentNeuralLayer) network.getLayer(1)).setReturnSequences(false);
        NNVector newInput = input;
        for (int i = 0; i < countChars; i++) {
            newInput = new NNVector(input.size() + 1);
            for (int j = 0; j < input.size(); j++) {
                newInput.set(j, input.get(j));
            }
            int max = network.query(new NNVector[]{input})[0].indexMaxElement();
            newInput.set(newInput.size() - 1, max);
            input = newInput;
        }
        ((RecurrentNeuralLayer) network.getLayer(1)).setReturnSequences(true);

        return newInput;
    }
//
//    public NNVector queryTest(NNVector input, int countChars) {
//        for (int i = network.size() - 1; i >= 0 ; i--) {
//            if(network.getLayer(i) instanceof RecurrentNeuralLayer){
//                ((RecurrentNeuralLayer) network.getLayer(i)).setReturnSequences(false).setReturnState(true);
//                break;
//            }
//        }
//        network.query(new NNVector[]{input});
//        for (int i = network.size() - 1; i >= 0 ; i--) {
//            if(network.getLayer(i) instanceof RecurrentNeuralLayer){
//                ((RecurrentNeuralLayer) network.getLayer(i)).setReturnOwnState(true);
//            }
//        }
//        NNVector output = new NNVector(input.size() + countChars);
//        int size = input.size();
//        for (int j = 0; j < input.size(); j++) {
//            output.set(j, input.get(j));
//        }
//        int max = network.getOutputs()[0].indexMaxElement();
//        input = new NNVector(1);
//        input.set(0, max);
//        output.set(size, max);
//        for (int i = 0; i < countChars; i++) {
//            network.query(new NNVector[]{input});
//            max = network.getOutputs()[0].indexMaxElement();
//            input = new NNVector(1);
//            input.set(0, max);
//            output.set(i + size, max);
//        }
//        for (int i = network.size() - 1; i >= 0 ; i--) {
//            if(network.getLayer(i) instanceof RecurrentNeuralLayer){
//                ((RecurrentNeuralLayer) network.getLayer(i)).setReturnOwnState(false);
//            }
//        }
//
//        return output;
//    }
}
