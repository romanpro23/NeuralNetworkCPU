package data.nlp;

import data.loaders.DataLoader1D;
import data.loaders.ImageData1D;
import data.network_train.NNData1D;
import lombok.SneakyThrows;
import nnarrays.NNVector;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Scanner;

public class EnUaTranslateLoader extends DataLoader1D {
    private int sizeEnVocabulary, sizeUaVocabulary;

    private LinkedHashMap<Integer, String> enWords;
    private LinkedHashMap<Integer, String> uaWords;

    private LinkedHashMap<String, Integer> enVocabulary;
    private LinkedHashMap<String, Integer> uaVocabulary;

    public EnUaTranslateLoader(int sizeEnVocabulary, int sizeUaVocabulary) {
        this.sizeEnVocabulary = sizeEnVocabulary;
        this.sizeUaVocabulary = sizeUaVocabulary;

        test = new ArrayList<>(0);
        train = new ArrayList<>(25000);

        loadData();
    }

    @SneakyThrows
    private void loadData() {
        Scanner scannerEnV = new Scanner(new File("D:/datasets/nlp/ua/en.txt"));
        Scanner scannerUaV = new Scanner(new File("D:/datasets/nlp/ua/ua.txt"));
        Scanner scanner = new Scanner(new File("D:/datasets/nlp/ua/en-ua.txt"));

        enVocabulary = new LinkedHashMap<>();
        uaVocabulary = new LinkedHashMap<>();
        enWords = new LinkedHashMap<>();
        uaWords = new LinkedHashMap<>();

        enVocabulary.put("<START>", 0);
        enWords.put(0, "<START>");
        uaVocabulary.put("<START>", 0);
        uaWords.put(0, "<START>");
        enVocabulary.put("<END>", 1);
        enWords.put(1, "<END>");
        enVocabulary.put("<UNK>", 2);
        enWords.put(2, "<UNK>");
        uaVocabulary.put("<END>", 1);
        uaWords.put(1, "<END>");
        uaVocabulary.put("<UNK>", 2);
        uaWords.put(2, "<UNK>");

        for (int i = 3; i < sizeEnVocabulary; i++) {
            String str = scannerEnV.nextLine();
            enVocabulary.put(str, i);
            enWords.put(i, str);
        }
        for (int i = 3; i < sizeUaVocabulary; i++) {
            String str = scannerUaV.nextLine();
            uaVocabulary.put(str, i);
            uaWords.put(i, str);
        }

        while (scanner.hasNextLine()) {
            String[] data = scanner.nextLine().split("\t");
            String[] enWords = data[0].split(" ");
            String[] uaWords = data[1].split(" ");

            NNVector input = getEnVector(enWords);
            NNVector output = getUaVector(uaWords);

            train.add(new ImageData1D(input, output));
        }

        Collections.shuffle(train);
    }

    public NNVector getUaVector(String[] uaWords){
        int index;
        NNVector output = new NNVector(uaWords.length + 1);
        for (int i = 0; i < uaWords.length; i++) {
            if (uaVocabulary.get(uaWords[i]) != null) {
                index = uaVocabulary.get(uaWords[i]);
            } else {
                index = 2;
            }
            output.set(i + 1, index);
        }

        return output;
    }

    public NNVector getEnVector(String[] enWords){
        NNVector input = new NNVector(enWords.length + 1);
        int index;
        for (int i = 0; i < enWords.length; i++) {
            if (enVocabulary.get(enWords[i]) != null) {
                index = enVocabulary.get(enWords[i]);
            } else {
                index = 2;
            }
            input.set(i + 1, index);
        }

        return input;
    }

    public String decodeUaString(NNVector input) {
        StringBuilder string = new StringBuilder();
        for (int i = 0; i < input.size(); i++) {
            if (input.get(i) == 1) {
                break;
            }
            string.append(uaWords.get((int) input.get(i)) + " ");
        }
        return string.toString();
    }
}
