package data.ciraf;

import data.loaders.DataLoader1D;
import data.loaders.ImageData1D;
import data.loaders.TransformData;
import nnarrays.NNVector;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

public class Ciraf100Loader1D extends DataLoader1D {
    private TransformData transformData;
    private float[] inputsData;
    private int trueNumb = -1;
    private byte[] bytes = new byte[3072];
    private byte[] byteNumb = new byte[2];

    private static FileInputStream scanner;
    private static FileInputStream scannerTest;

    private void loadTrainFilesWithNumber() {
        try {
            scanner = new FileInputStream("D:/datasets/ciraf/train.bin");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void loadTestFilesWithNumber() {
        try {
            scannerTest = new FileInputStream("D:/datasets/ciraf/test.bin");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Ciraf100Loader1D() {
        this(new TransformData.Sigmoid());
    }

    public Ciraf100Loader1D(TransformData transform) {
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
        loadTrainFilesWithNumber();
        for (int i = 0; i < 50000; i++) {
            if (scanner.available() >= 3072) {
                scanner.read(byteNumb);
                scanner.read(bytes);

                trueNumb = byteNumb[0];
                generateInput();
                NNVector output = new NNVector(100);
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
                NNVector output = new NNVector(100);
                output.set(trueNumb, 1);

                test.add(new ImageData1D(new NNVector(inputsData), output));;
            }
        }
        Collections.shuffle(test);
    }

    private void generateInput() {
        inputsData = new float[3072];
        for (int i = 0; i < 3072; i++) {
            inputsData[i] = transformData.transform(bytes[i]);
        }
    }
}