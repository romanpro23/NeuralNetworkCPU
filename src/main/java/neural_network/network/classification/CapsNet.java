package neural_network.network.classification;

import neural_network.layers.capsule.CapsuleLayer;
import neural_network.layers.capsule.DigitCapsuleLayer;
import neural_network.layers.capsule.MaskLayer;
import neural_network.network.NeuralNetwork;
import nnarrays.NNArray;
import nnarrays.NNArrays;

public class CapsNet {
    private NeuralNetwork classificator;
    private NeuralNetwork decoder;

    private MaskLayer maskLayer;
    private CapsuleLayer layer;

    public CapsNet(NeuralNetwork classificator, NeuralNetwork decoder){
        this.classificator = classificator;
        this.decoder = decoder;

        this.maskLayer = new MaskLayer();

        for (int i = classificator.getLayers().size() - 1; i > 0; i--) {
            if(classificator.getLayer(i) instanceof CapsuleLayer){
                layer = (CapsuleLayer) classificator.getLayer(i);
                maskLayer.initialize(layer.size());
                layer.addNextLayer(maskLayer);
                break;
            }
        }
    }

    public float train(NNArray[] input, NNArray[] output){
        return train(input, output, 1);
    }

    public float train(NNArray[] input, NNArray[] output, float lambda){
        float accuracy = 0;
        /*classificator.queryTrain(input);
        maskLayer.generateOutput(layer.getOutput(), output);
        accuracy = decoder.train(maskLayer.getOutput(), input);
        maskLayer.generateError(decoder.getError());
        if(lambda != 1) {
            NNArrays.mul(maskLayer.getErrorNL(), lambda);
        }
        accuracy += classificator.trainOutput(output);*/


        return accuracy;
    }

    public NNArray[] query(NNArray[] input){
        return classificator.query(input);
    }

    public NNArray[] queryDecoder(NNArray[] input, NNArray[] label){
        classificator.query(input);
        maskLayer.generateOutput(layer.getOutput(), label);
        return decoder.query(maskLayer.getOutput());
    }

    public NNArray[] queryDecoder(NNArray[] input){
        classificator.query(input);
        return decoder.query(layer.getOutput());
    }
}
