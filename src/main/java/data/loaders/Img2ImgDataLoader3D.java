package data.loaders;

import data.network_train.NNData3D;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.util.ArrayList;
import java.util.Collections;

public abstract class Img2ImgDataLoader3D {
    protected ArrayList<NNTensor> trainA;
    protected ArrayList<NNTensor> testA;

    protected ArrayList<NNTensor> trainB;
    protected ArrayList<NNTensor> testB;

    private int curTrainA = 0, curTestA = 0, curTrainB = 0, curTestB = 0;

    protected void reloadTrainData(){};
    protected void reloadTestData(){};

    public NNTensor[] getNextTrainAData(int sizeBatch) {
        int size = sizeBatch;
        if (curTrainA + sizeBatch >= trainA.size()) {
            size = trainA.size() - curTrainA;
        }

        NNTensor[] input = new NNTensor[size];

        for (int i = 0; i < size; i++) {
            input[i] = trainA.get(curTrainA);

            curTrainA++;
        }

        if(size != sizeBatch || curTrainA == trainA.size()){
            curTrainA = 0;
            reloadTrainData();
            Collections.shuffle(trainA);
        }

        return input;
    }

    public NNTensor[] getNextTrainBData(int sizeBatch) {
        int size = sizeBatch;
        if (curTrainB + sizeBatch >= trainB.size()) {
            size = trainB.size() - curTrainB;
        }

        NNTensor[] input = new NNTensor[size];

        for (int i = 0; i < size; i++) {
            input[i] = trainB.get(curTrainB);

            curTrainB++;
        }

        if(size != sizeBatch || curTrainB == trainB.size()){
            curTrainB = 0;
            reloadTrainData();
            Collections.shuffle(trainB);
        }

        return input;
    }

    public NNTensor[] getNextTestAData(int sizeBatch) {
        int size = sizeBatch;
        if (curTestA + sizeBatch >= testA.size()) {
            size = testA.size() - curTestA;
        }

        NNTensor[] input = new NNTensor[size];

        for (int i = 0; i < size; i++) {
            input[i] = testA.get(curTestA);

            curTestA++;
        }

        if(size != sizeBatch || curTestA == testA.size()){
            curTestA = 0;
            reloadTestData();
            Collections.shuffle(testA);
        }

        return input;
    }

    public NNTensor[] getNextTestBData(int sizeBatch) {
        int size = sizeBatch;
        if (curTestB + sizeBatch >= testB.size()) {
            size = testB.size() - curTestB;
        }

        NNTensor[] input = new NNTensor[size];

        for (int i = 0; i < size; i++) {
            input[i] = testB.get(curTestB);

            curTestB++;
        }

        if(size != sizeBatch || curTestB == testB.size()){
            curTestB = 0;
            reloadTestData();
            Collections.shuffle(testB);
        }

        return input;
    }
}
