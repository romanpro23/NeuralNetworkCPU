package test.classification.imagenet;

import data.ImageCreator;
import data.imageNet.*;
import neural_network.network.NeuralNetwork;
import nnarrays.NNArray;
import nnarrays.NNArrays;
import nnarrays.NNTensor;

import java.io.File;
import java.util.Arrays;
import java.util.Scanner;

public class TestImageNet250 {
    public static void main(String[] args) throws Exception {
        NeuralNetwork network = NeuralNetwork
                .read(new Scanner(new File("D:/NetworkTest/Imagenet/alexnet.txt")))
                .create();

        String[] strings = new String[]{
                "cat.jpg",
                "cat2.jpg",
                "rabbit.jpg",
                "orange.jpg",
                "orange2.png",
                "tiger.jpg",
                "tiger2.jpg",
                "tiger3.jpg",
                "labrador.jpg"
        };

        NNTensor[] input = new NNTensor[strings.length];
        for (int i = 0; i < strings.length; i++) {
            input[i] = ImageCreator.loadImage("D:/NetworkTest/Photo/" + strings[i], 64);
        }

        NNArray[] result = network.query(input);
        for (int i = 0; i < result.length; i++) {
            System.out.println(strings[i] + " - " + Arrays.toString(ImageNet250Loader3D.getLabelVal(result[i].indexMaxElement(5)) ));
        }
    }
}
