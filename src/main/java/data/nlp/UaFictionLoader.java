package data.nlp;

import data.loaders.DataLoader1D;
import data.loaders.ImageData1D;
import lombok.SneakyThrows;
import nnarrays.NNVector;

import java.io.File;
import java.util.*;

public class UaFictionLoader extends DataLoader1D {
    private final int sizeUaVocabulary;

    private LinkedHashMap<Integer, String> uaWords;
    private LinkedHashMap<String, Integer> uaVocabulary;

    private int maxLength;
    private boolean addPaddingOnStart;

    public UaFictionLoader(int sizeUaVocabulary) {
        this(sizeUaVocabulary, 0, false);
    }

    public UaFictionLoader(int sizeUaVocabulary, int maxLength, boolean addPaddingOnStart) {
        this.sizeUaVocabulary = sizeUaVocabulary;

        test = new ArrayList<>(0);
        train = new ArrayList<>(25000);

        if(maxLength != 0){
            this.maxLength = maxLength;
            this.addPaddingOnStart = addPaddingOnStart;
        }

        loadData();
    }

    @SneakyThrows
    private void loadData() {
        Scanner scannerUaV = new Scanner(new File("D:\\datasets\\nlp\\Ukrainian\\ua_voc.txt"));
        Scanner scanner = new Scanner(new File("D:\\datasets\\nlp\\Ukrainian\\ua_fiction.txt"));

        uaVocabulary = new LinkedHashMap<>();
        uaWords = new LinkedHashMap<>();

        uaVocabulary.put(NLP._SOS, NLP.SOS);
        uaWords.put(NLP.SOS, NLP._SOS);
        uaVocabulary.put(NLP._EOS, NLP.EOS);
        uaWords.put(NLP.EOS, NLP._EOS);
        uaVocabulary.put(NLP._UNK, NLP.UNK);
        uaWords.put(NLP.UNK, NLP._UNK);
        uaVocabulary.put(NLP._PAD, NLP.PAD);
        uaWords.put(NLP.PAD, NLP._PAD);

        for (int i = 5; i < sizeUaVocabulary; i++) {
            String str = scannerUaV.nextLine();
            uaVocabulary.put(str, i);
            uaWords.put(i, str);
        }

        while (scanner.hasNextLine()) {
            String[] data = scanner.nextLine().split("[ ,.:;!?/()\"*%»«]");

            if(data.length >= 3) {
                int inputSize = data.length / 2 + 1;
                String[] dataIn = Arrays.copyOfRange(data, 0, inputSize);
                String[] dataOut = Arrays.copyOfRange(data, inputSize, data.length);
                NNVector input = getUaVector(dataIn);
                NNVector output = getUaVector(dataOut);

                train.add(new ImageData1D(input, output));
            }
        }

        Collections.shuffle(train);
    }

    public NNVector getUaVector(String[] uaWords){
        NNVector input;
        int index;
        int start = 1;
        int end = uaWords.length + 1;
        if(maxLength != 0 ){
            if(addPaddingOnStart) {
                start = maxLength - uaWords.length - 1;
                end = maxLength - 1;
                input = new NNVector(maxLength);
                input.set(0, NLP.SOS);
                for (int i = 1; i < start; i++) {
                    input.set(i, NLP.PAD);
                }
            } else {
                end = maxLength - uaWords.length - 1;
                input = new NNVector(maxLength);
                input.set(end, NLP.SOS);
                for (int i = end + 1; i < maxLength; i++) {
                    input.set(i, NLP.PAD);
                }
            }
        } else {
            input = new NNVector(uaWords.length + 2);
            input.set(0, NLP.SOS);
        }

        for (int i = start; i < end; i++) {
            Integer data = uaVocabulary.get(uaWords[i-start]);
            index = Objects.requireNonNullElse(data, NLP.UNK);
            input.set(i, index);
        }
        input.set(end, NLP.EOS);

        return input;
    }

    public String decodeUaString(NNVector input) {
        StringBuilder string = new StringBuilder();
        for (int i = 0; i < input.size(); i++) {
            if (input.get(i) == NLP.EOS) {
                break;
            }
            string.append(uaWords.get((int) input.get(i)) + " ");
        }
        return string.toString();
    }
}
