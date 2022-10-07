package data.imdb;

import data.loaders.DataLoader1D;
import data.loaders.ImageData1D;
import data.network_train.NNData1D;
import nnarrays.NNVector;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Scanner;

public class IMDBLoader1D extends DataLoader1D {
    private NNVector trueAns = new NNVector(new float[]{1});
    private NNVector falseAns = new NNVector(new float[]{0});

    public IMDBLoader1D() {
        test = new ArrayList<>(25000);
        train = new ArrayList<>(25000);

        try {
            loadData();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    private void loadData() throws FileNotFoundException {
        Scanner train = new Scanner(new File("D:/datasets/imdb/imdb_train.txt"));
        Scanner trainLable = new Scanner(new File("D:/datasets/imdb/train_valid.txt"));
        Scanner test = new Scanner(new File("D:/datasets/imdb/imdb_test.txt"));
        Scanner testLable = new Scanner(new File("D:/datasets/imdb/test_valid.txt"));

        for (int i = 0; i < 25000; i++) {
            int[] arrTrain = Arrays.stream(train.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
            int lableTrain = Integer.parseInt(trainLable.nextLine());
            if(lableTrain == 1){
                this.train.add(new ImageData1D(new NNVector(arrTrain), trueAns));
            } else {
                this.train.add(new ImageData1D(new NNVector(arrTrain), falseAns));
            }

            int[] arrTest = Arrays.stream(test.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
            int lableTest = Integer.parseInt(testLable.nextLine());
            if(lableTest == 1){
                this.test.add(new ImageData1D(new NNVector(arrTest), trueAns));
            } else {
                this.test.add(new ImageData1D(new NNVector(arrTest), falseAns));
            }
        }
    }
}