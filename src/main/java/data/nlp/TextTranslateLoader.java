package data.nlp;

import com.opencsv.CSVReader;
import lombok.SneakyThrows;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class TextTranslateLoader {

    public void rewrite(Scanner scanner, String path) throws IOException {
        String[] translate;

        FileWriter writer = new FileWriter(path);

        while (scanner.hasNextLine()) {
            translate = scanner.nextLine().split("\t");
            translate[0] = translate[0].toLowerCase();
            for (int i = 0; i < translate[0].length(); i++) {
                char ch = translate[0].charAt(i);
                if((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == ' ' || ch == '\'' || ch == '-'){
                    writer.write(ch);
                }
            }
            writer.write("\t");
            translate[1] = translate[1].toLowerCase();
            for (int i = 0; i < translate[1].length(); i++) {
                char ch = translate[1].charAt(i);
                if((ch >= 1072 && ch <= 1103) ||(ch >= 1108 && ch <= 1111) || (ch >= '0' && ch <= '9') || ch == ' ' || ch == '\'' || ch == '-'){
                    writer.write(ch);
                }
            }
            writer.write("\n");
            writer.flush();
        }
    }

    @SneakyThrows
    public void rewrite(String pathInput, String pathTxt) throws IOException {
        Scanner reader = new Scanner(new File(pathInput));
        FileWriter writer = new FileWriter(pathTxt);

        for (int i = 0; i < 158915; i++) {
            String text = reader.nextLine().split("\\|")[2].toLowerCase();
            System.out.println(text);
            for (int t = 0; t < text.length(); t++) {
                char ch = text.charAt(t);
                if((ch >= 97 && ch <= 123) || (ch >= '0' && ch <= '9') || ch == ' ' || ch == '\'' || ch == '-'){
                    writer.write(ch);
                }
            }
            writer.write("\n");
            if(i % 5 == 4){
                writer.write("\n");
            }
            writer.flush();
        }
    }

    public void createVocabularyWords(Scanner scanner, String path) throws IOException {
        String[] translate;
        String[] englishWord;
        String[] translateWord;

        LinkedHashMap<String, Integer> englishWords = new LinkedHashMap<>();
        LinkedHashMap<String, Integer> translateWords = new LinkedHashMap<>();

        while (scanner.hasNextLine()) {
            translate = scanner.nextLine().split("\t");
            //[ ,.:;!?/()"*%»«]
            englishWord = translate[0].split(" ");
            translateWord = translate[1].split(" ");

            for (String s : englishWord) {
                if (!s.equals(""))
                    if (englishWords.containsKey(s)) {
                        Integer val = englishWords.get(s);
                        val++;
                        englishWords.put(s, val);
                    } else {
                        englishWords.put(s, 1);
                    }
            }

            for (String s : translateWord) {
                if (!s.equals(""))
                    if (translateWords.containsKey(s)) {
                        Integer val = translateWords.get(s);
                        val++;
                        translateWords.put(s, val);
                    } else {
                        translateWords.put(s, 1);
                    }
            }
        }

        englishWords = sortHashMapByValues(englishWords);

        List<String> enVoc = new ArrayList<>(englishWords.keySet());

        FileWriter writerEn = new FileWriter(path + "en.txt");
        System.out.println(enVoc.size());
        for (int i = enVoc.size() - 1; i >= 0; i--) {
            writerEn.write(enVoc.get(i) + "\n");
            writerEn.flush();
        }

        FileWriter writerTransl = new FileWriter(path + "transl.txt");
        List<Integer> mapValues = new ArrayList<>(translateWords.values());
        List<String> mapKeys = new ArrayList<>(translateWords.keySet());

        while(!mapKeys.isEmpty()){
            int max = mapValues.get(0), indexMax = 0;
            for (int i = 0; i < mapValues.size(); i++) {
                if(mapValues.get(i) > max){
                    max = mapValues.get(i);
                    indexMax = i;
                }
            }
            writerTransl.write(mapKeys.get(indexMax) + "\n");
            mapKeys.remove(indexMax);
            mapValues.remove(indexMax);
            writerTransl.flush();
        }
    }

    public void createOneVocabularyWords(Scanner scanner, String path) throws IOException {
        String[] word;

        LinkedHashMap<String, Integer> words = new LinkedHashMap<>();

        int ind = 0;
        while (scanner.hasNextLine()) {
            word = scanner.nextLine().split("[ ,.:;!?/()\"*%»«]");

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

            if(ind % 10000 == 0){
                System.out.println(ind);
            }
            ind++;
        }

//        List<String> mapKeys = new ArrayList<>(words.keySet());
//        System.out.println(mapKeys.size());
//        for (int i = 0; i < mapKeys.size(); i++) {
//            if(words.get(mapKeys.get(i)) <= 20){
//                words.remove(mapKeys.get(i));
//            }
//        }

        words = sortHashMapByValues(words);

        List<String> voc = new ArrayList<>(words.keySet());

        FileWriter writer = new FileWriter(path);
        System.out.println(voc.size());
        for (int i = voc.size() - 1; i >= 0; i--) {
            writer.write(voc.get(i) + "\n");
            writer.flush();
        }
    }

    private LinkedHashMap<String, Integer> sortHashMapByValues(HashMap<String, Integer> passedMap) {
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

}
