package test.nlp;

import data.nlp.EnUaTranslateLoader;
import data.nlp.TextTranslateLoader;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

import static test.speechtotext.PositionLoader.fileNameWithOutExt;

public class TestTextTranslateLoader {
    public static void main(String[] args) throws IOException {
//        TextTranslateLoader loader = new TextTranslateLoader("D:/datasets/nlp/ukr.txt");
//        TextTranslateLoader loader = new TextTranslateLoader("D:/datasets/nlp/en-ua.txt");
//
//        loader.rewrite("D:/datasets/nlp/en-ua.txt");
//        loader.createVocabularyWords("D:/datasets/nlp/");

//        EnUaTranslateLoader loader = new EnUaTranslateLoader(10500, 29000);
//
//        System.out.println(Arrays.toString(loader.getNextTrainData(1).getInput()[0].getData()));

        TextTranslateLoader loader = new TextTranslateLoader();

        String fileName = "C:/Levani/cv-corpus-15.0-2023-09-08/ka/validated.tsv";
        File file = new File(fileName);
        ArrayList<String[]> data = tsvr(file);

        File folder = new File("C:/Levani/SpeechToSpeech-0.013/images/");
        File[] listOfFiles = folder.listFiles();

        ArrayList<String> Listdata = new ArrayList<>();

        int dd = 0;
        for (File afile : listOfFiles) {
            if (afile.isFile()) {
                StringBuilder label = new StringBuilder();
                for (String[] datum : data) {
                    if (Objects.equals(fileNameWithOutExt(datum[1]), fileNameWithOutExt(afile.getName()))) {
                        label = new StringBuilder(datum[2]);
                        break;
                    }
                }
                if (!Objects.equals(label.toString(), "")) {
                    Listdata.add(label.toString());
                }
            }

            //if (dd == 299) {
            //    break;
            //}

            if (dd % 100 == 0) {
                System.out.println(dd);
            }
            dd++;
        }

        LinkedHashMap<String, Integer> words = new LinkedHashMap<>();

        int ind = 0;

        String[] word = null;

        for (String ss : Listdata) {
            word = ss.split("[ ,.:;!?/()\"*%��–„“-]");

            for (String s : word) {
                if (!s.equals(""))
                    if (words.containsKey(s)) {
                        Integer val = words.get(s);
                        val++;
                        words.put(s, val);
                    } else {
                        words.put(s, 1);
                    }
            }

            if (ind % 10000 == 0) {
                System.out.println(ind);
            }
            ind++;
        }

        words = sortHashMapByValues(words);

        List<String> voc = new ArrayList<>(words.keySet());

        FileWriter writer = new FileWriter("C:\\Levani\\vocabulary.txt");
        System.out.println(voc.size());
        for (int i = voc.size() - 1; i >= 0; i--) {
            writer.write(voc.get(i) + "\n");
            writer.flush();
        }


        //loader.rewrite("C:\\Levani\\1.txt", "C:\\Levani\\2.txt");
        //loader.createOneVocabularyWords(new Scanner(new File("C:\\Levani\\1.txt")), "C:\\Levani\\vocabulary.txt");
    }

    private static LinkedHashMap<String, Integer> sortHashMapByValues(HashMap<String, Integer> passedMap) {
        List<String> mapKeys = new ArrayList<>(passedMap.keySet());
        List<Integer> mapValues = new ArrayList<>(passedMap.values());

        Collections.sort(mapValues);
        Collections.sort(mapKeys);

        LinkedHashMap<String, Integer> sortedMap = new LinkedHashMap<>();
        Iterator<Integer> valueIt = mapValues.iterator();

        while (valueIt.hasNext()) {
            Integer val = valueIt.next();
            Iterator<String> keyIt = mapKeys.iterator();

            while (keyIt.hasNext()) {
                String key = keyIt.next();
                Integer comp1 = passedMap.get(key);
                Integer comp2 = val;

                if (comp1.equals(comp2)) {
                    keyIt.remove();
                    sortedMap.put(key, val);
                    break;
                }
            }
        }

        return sortedMap;
    }

    public static ArrayList<String[]> tsvr(File test2) {
        ArrayList<String[]> Data = new ArrayList<>(); //initializing a new ArrayList out of String[]'s
        try (BufferedReader TSVReader = new BufferedReader(new InputStreamReader(new FileInputStream(test2), "UTF-8"))) {
            String line = null;
            while ((line = TSVReader.readLine()) != null) {
                String[] lineItems = line.split("\t"); //splitting the line and adding its items in String[]
                Data.add(lineItems); //adding the splitted line array to the ArrayList
            }
        } catch (Exception e) {
            System.out.println("Something went wrong");
        }
        return Data;
    }
}
