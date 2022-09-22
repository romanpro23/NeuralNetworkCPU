package data.mnist;

import nnarrays.NNVector;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class MNISTLoader1D extends DataLoader1D {
    private final String PATH_TO_MNIST_DATASET = "D:/datasets/mnist_batch/";

    private final TransformData transformData;
    private float[] inputsData;
    private int trueNumb = -1;
    private final byte[] bytes = new byte[784];
    private final byte[] byteNumb = new byte[1];

    private final ArrayList<ImageData1D> train;
    private final ArrayList<ImageData1D> test;

    private int curTrain = 0;
    private int curTest = 0;

    private static FileInputStream scanner;
    private static FileInputStream scannerNumb;
    private static FileInputStream scannerTest;
    private static FileInputStream scannerNumbTest;

    private BatchMNIST batchMNIST;

    private void loadTrainFilesWithNumber() {
        try {
            scanner = new FileInputStream(PATH_TO_MNIST_DATASET + batchMNIST.getTrainFile());
            scannerNumb = new FileInputStream(PATH_TO_MNIST_DATASET + batchMNIST.getTrainFileMark());

            scanner.skip(16);
            scannerNumb.skip(8);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void loadTestFilesWithNumber() {
        try {
            scannerTest = new FileInputStream(PATH_TO_MNIST_DATASET + batchMNIST.getTestFile());
            scannerNumbTest = new FileInputStream(PATH_TO_MNIST_DATASET + batchMNIST.getTestFileMark());

            scannerTest.skip(16);
            scannerNumbTest.skip(8);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public MNISTLoader1D() {
        this(BatchMNIST.MNIST);
    }

    public MNISTLoader1D(BatchMNIST batchMNIST) {
        this(batchMNIST, new TransformData.Sigmoid());
    }

    public MNISTLoader1D(BatchMNIST batchMNIST, TransformData transform) {
        this.batchMNIST = batchMNIST;
        this.transformData = transform;

        loadTrainFilesWithNumber();
        loadTestFilesWithNumber();

        train = new ArrayList<>(batchMNIST.getSizeTrain());
        test = new ArrayList<>(batchMNIST.getSizeTest());

        try {
            loadTrainData();
            loadTestData();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void loadTrainData() throws IOException {
        for (int i = 0; i < batchMNIST.getSizeTrain(); i++) {
            if (scannerNumb.available() > 0) {
                scannerNumb.read(byteNumb);
                scanner.read(bytes);
                trueNumb = byteNumb[0];
                generateInput();
                NNVector output = new NNVector(batchMNIST.getCountClass());
                output.set(trueNumb, 1);
                train.add(new ImageData1D(new NNVector(inputsData), output));
            }
        }
        Collections.shuffle(train);
    }

    private void loadTestData() throws IOException {
        for (int i = 0; i < batchMNIST.getSizeTest(); i++) {
            if (scannerNumbTest.available() > 0) {
                scannerNumbTest.read(byteNumb);
                scannerTest.read(bytes);
                trueNumb = byteNumb[0];
                generateInput();
                NNVector output = new NNVector(batchMNIST.getCountClass());
                output.set(trueNumb, 1);
                test.add(new ImageData1D(new NNVector(inputsData), output));
            }
        }
        Collections.shuffle(test);
    }

    public NNData1D getNextTrainData(int sizeBatch) {
        int size = sizeBatch;
        if (curTrain + sizeBatch >= batchMNIST.getSizeTrain()) {
            size = batchMNIST.getSizeTrain() - curTrain;
        }

        NNVector[] input = new NNVector[size];
        NNVector[] output = new NNVector[size];

        for (int i = 0; i < size; i++) {
            input[i] = train.get(curTrain).getInputs();
            output[i] = train.get(curTrain).getOutputs();

            curTrain++;
        }

        if(size != sizeBatch || curTrain == batchMNIST.getSizeTrain()){
            curTrain = 0;
            Collections.shuffle(train);
        }

        return new NNData1D(input, output);
    }

    public NNData1D getNextTestData(int sizeBatch) {
        int size = sizeBatch;
        if (curTest + sizeBatch >= batchMNIST.getSizeTest()) {
            size = batchMNIST.getSizeTest() - curTest;
        }

        NNVector[] input = new NNVector[size];
        NNVector[] output = new NNVector[size];

        for (int i = 0; i < size; i++) {
            input[i] = test.get(curTest).getInputs();
            output[i] = test.get(curTest).getOutputs();

            curTest++;
        }

        if(size != sizeBatch || curTest == batchMNIST.getSizeTest()){
            curTest = 0;
            Collections.shuffle(test);
        }

        return new NNData1D(input, output);
    }

    private void generateInput() {
        inputsData = new float[784];
        for (int i = 0; i < 784; i++) {
            inputsData[i] = transformData.transform(bytes[i]);
        }
    }
}