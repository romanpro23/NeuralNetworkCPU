package data.ciraf;

import data.loaders.*;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class Ciraf10Loader1D extends DataLoader1D {
    private TransformData transformData;
    private float[] inputsData;
    private int trueNumb = -1;
    private byte[] bytes = new byte[3072];
    private byte[] byteNumb = new byte[1];

    private static FileInputStream scanner;
    private static FileInputStream scannerTest;

    private void loadTrainFilesWithNumber(int numbBatch) {
        try {
            scanner = new FileInputStream("D:/datasets/ciraf/data_batch_" + numbBatch + ".bin");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void loadTestFilesWithNumber() {
        try {
            scannerTest = new FileInputStream("D:/datasets/ciraf/test_batch.bin");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Ciraf10Loader1D() {
        this(new TransformData.Sigmoid());
    }

    public Ciraf10Loader1D(TransformData transform) {
        this.transformData = transform;

        train = new ArrayList<>(50000);
        test = new ArrayList<>(10000);

        try {
            loadTrainData();
            loadTestData();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void loadTrainData() throws IOException {
        int n = 1;
        for (int i = 0; i < 50000; i++) {
            if(i % 10000 == 0) {
                loadTrainFilesWithNumber(n);
                n++;
            }
            if (scanner.available() >= 3072) {
                scanner.read(byteNumb);
                scanner.read(bytes);

                trueNumb = byteNumb[0];
                generateInput();
                NNVector output = new NNVector(10);
                output.set(trueNumb, 1);

                train.add(new ImageData1D(new NNVector(inputsData), output));
            }

        }
        Collections.shuffle(train);
    }

    private void loadTestData() throws IOException {
        loadTestFilesWithNumber();
        for (int i = 0; i < 10000; i++) {
            if (scannerTest.available() >= 3072) {
                scannerTest.read(byteNumb);
                scannerTest.read(bytes);
                trueNumb = byteNumb[0];
                generateInput();
                NNVector output = new NNVector(10);
                output.set(trueNumb, 1);

                test.add(new ImageData1D(new NNVector(inputsData), output));;
            }
        }
        Collections.shuffle(test);
    }

    private void generateInput() {
        inputsData = new float[3072];
        for (int i = 0; i < 1024; i++) {
            inputsData[i] = transformData.transformR(bytes[i]);
        }
        for (int i = 1024; i < 2048; i++) {
            inputsData[i] = transformData.transformG(bytes[i]);
        }
        for (int i = 2048; i < 3072; i++) {
            inputsData[i] = transformData.transformB(bytes[i]);
        }
    }
}