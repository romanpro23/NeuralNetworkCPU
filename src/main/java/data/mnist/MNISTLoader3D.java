package data.mnist;

import data.loaders.DataLoader3D;
import data.loaders.ImageData3D;
import data.loaders.TransformData;
import data.network_train.NNData3D;
import nnarrays.NNMatrix;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class MNISTLoader3D extends DataLoader3D {
    private final String PATH_TO_MNIST_DATASET = "D:/datasets/mnist_batch/";

    private final TransformData transformData;
    private float[] inputsData;
    private int trueNumb = -1;
    private final byte[] bytes = new byte[784];
    private final byte[] byteNumb = new byte[1];

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

    public MNISTLoader3D() {
        this(BatchMNIST.MNIST);
    }

    public MNISTLoader3D(BatchMNIST batchMNIST) {
        this(batchMNIST, new TransformData.Sigmoid());
    }

    public MNISTLoader3D(BatchMNIST batchMNIST, TransformData transform) {
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
                train.add(new ImageData3D(new NNTensor(28, 28, 1, inputsData), output));
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
                test.add(new ImageData3D(new NNTensor(28, 28, 1, inputsData), output));
            }
        }
        Collections.shuffle(test);
    }

    private void generateInput() {
        inputsData = new float[784];
        for (int i = 0; i < 784; i++) {
            inputsData[i] = transformData.transform(bytes[i]);
        }
        inputsData = new NNMatrix(28,28, inputsData).transpose().getData();
    }
}