package test.speechtotext;

import data.loaders.*;
import data.network_train.NNData2D;
import data.nlp.NLP;
import jcuda.Pointer;
import jcuda.Sizeof;
import nnarrays.*;
import utilities.Use;
import data.nlp.UaFictionLoader;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static nnarrays.NNArray.bFloat16ToFloat;
import static nnarrays.NNArray.floatToBFloat16;
import static utilities.Use.*;

public class PositionLoader extends DataLoader2D {
    private LinkedHashMap<Integer, Character> uaChars;
    private LinkedHashMap<Integer, String> uaWords;
    private LinkedHashMap<Character, Integer> codeUaChars;
    private LinkedHashMap<String, Integer> uaVocabulary;

    private int maxLength = 0;
    private boolean addPaddingOnStart;

    float constant = 0.1f;//sizeVocabulary / 10.0f;

    public PositionLoader(int countChars) throws Exception {

        train = new ArrayList<>();
        test = new ArrayList<>();

        if (WorkingWithCharacter) {
            uaChars = new LinkedHashMap<>();
            codeUaChars = new LinkedHashMap<>();

            try {
                Scanner scanner = new Scanner(new File("C:/Levani/NeuralNetworkCPU/data/ka_chars.txt"), StandardCharsets.UTF_8);
                uaChars.put(NLP.UNK, NLP._UNK_CHAR);
                codeUaChars.put(NLP._UNK_CHAR, NLP.UNK);
                uaChars.put(NLP.PAD, ' ');
                codeUaChars.put(' ', NLP.PAD);

                int key = 3;
                for (int i = 0; i < countChars; i++) {
                    Character str = scanner.nextLine().charAt(0);
                    uaChars.put(key, str);
                    codeUaChars.put(str, key);
                    key += 1;
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        else
        {
            Scanner scannerUaV = null;
            try {
                scannerUaV = new Scanner(new File("C:\\Levani\\vocabulary.txt"));
            } catch (FileNotFoundException e) {
                throw new RuntimeException(e);
            }

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

            for (int i = 5; i < sizeVocabulary; i++) {
                String str = scannerUaV.nextLine();
                uaVocabulary.put(str, i);
                uaWords.put(i, str);
            }
        }

        reloadTrainData();
    }

    @Override
    public void reloadTrainData() {

        String fileName = "C:/Levani/cv-corpus-15.0-2023-09-08/ka/validated.tsv";
        File file = new File(fileName);
        ArrayList<String[]> data = tsvr(file);

        File folder = new File("C:/Levani/SpeechToSpeech-0.013/images/");
        File[] listOfFiles = folder.listFiles();

        TransformData transformData = new TransformData.Sigmoid();

        int wwq = 0;
        for (File afile : listOfFiles) {
            if (afile.isFile()) {
                Use.CPU = true;
                BufferedImage img = null;
                try {
                    img = ImageIO.read(new File(folder.toPath() + "\\" + afile.getName()));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

                int width = img.getWidth();
                int height = img.getHeight();

                int size_width = 480;

                if (width > size_width) {
                    try {
                        //throw new Exception("!!!: " + width);
                        System.out.println("!!!: " + width);
                        continue;
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }

                NNMatrix inputsData = new NNMatrix(size_width, 24,false);

                Use.GPU = false;

                /*for (int xx = 0; xx < width; xx++) {
                    for (int yy = 0; yy < height; yy++) {
                        Color color = new Color(img.getRGB(xx, yy));
                        inputsData.set(yy, width - xx - 1, transformData.transformR(color.getRed()));
                    }
                }*/

                for (int xx = 0; xx < width; xx++) {
                    for (int yy = 0; yy < height; yy++) {
                        Color color = new Color(img.getRGB(xx, yy));
                        inputsData.set(xx, yy, transformData.transformR(color.getRed()));
                    }
                }

                Use.GPU = true;

                StringBuilder label = new StringBuilder();
                for (String[] datum : data) {
                    if (Objects.equals(fileNameWithOutExt(datum[1]), fileNameWithOutExt(afile.getName()))) {
                        label = new StringBuilder(datum[2]);
                        break;
                    }
                }

                if (!Objects.equals(label.toString(), "")) {
                    /*int c = WordCount;
                    //label = label.substring(0, c);//!!!
                    int sss = c - label.length();
                    if (sss < 0) {
                        try {
                            //System.out.println("!!!: " + sss);
                            //continue;
                            throw new Exception("!!!: " + sss);
                        } catch (Exception e) {
                            throw new RuntimeException(e);
                        }
                    }*/
                    String labelNew = label.toString();
                    /*for (int ww = 0; ww < sss; ww++) {
                        label.append("_");
                    }*/

                    NNVector output_;
                    NNVector output;
                    if (WorkingWithCharacter) {
                        output = codeString(label.toString(), false);
                        //output = NNArrays.toHotVector(output_, sizeVocabulary, WordCount);
                    }
                    else {
                        output_ = new NNVector(WordCount, false);

                        String[] data_ = labelNew.split("[ ,.:;!?/()\"*%��–„“-]");
                        output = getUaVector(data_, output_);
                    }

                    NNMatrix inputsDataNew = new NNMatrix(inputsData.getRow(), inputsData.getColumn(), inputsData.getData(), inputsData.getSdata(), true);//true

                    inputsDataNew.ClearCpuData();
                    inputsData = null;
                    inputsDataNew.ClearCpuData();

                    codeStringNew(output, false);
                    output.ClearCpuData();

                    Use.CPU = false;

                    train.add(new ImageData2D(inputsDataNew, output));
                    //test.add(new ImageData2D(inputsDataNew, output));

                    if (wwq % 100 == 0) {
                        System.out.println(wwq);
                    }

                    if (wwq == 299) {
                        return;
                    }

                    wwq++;
                }

                Use.CPU = false;
            }
        }
    }

    public NNVector getUaVector(String[] uaWords, NNVector input){
        //NNVector input;

        for (int i = 0; i < input.size(); i++) {
            input.set(i, NLP.EOS / constant/* + 1*/);
        }

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
            //input = new NNVector(uaWords.length + 2);
            input.set(0, NLP.SOS / constant/* + 1*/);
        }

        for (int i = start; i < end; i++) {
            Integer data = uaVocabulary.get(uaWords[i-start]);
            index = Objects.requireNonNullElse(data, NLP.UNK);
            input.set(i, ((float) index) / constant/* + 1*/);
        }
        input.set(end, NLP.EOS / constant/* + 1*/);

        return input;/*NNArrays.toHotVector(input, sizeVocabulary, WordCount);*///!!!!
    }

    public NNVector codeString(String text, boolean TYPE) {
        char[] chars = text.toCharArray();

        NNVector input = new NNVector(WordCount, TYPE);

        for (int i = 0; i < chars.length; i++) {
            float value = ((float) codeUaChars.get(chars[i]) / constant/* + 1*/);
            input.set(i, value);
        }

        return input;
    }

    public void codeStringNew(NNMatrix v, boolean TYPE) {
        cudaMemcpy(v.getData_gpu(), Pointer.to(v.getData()), (long) Sizeof.FLOAT * v.size(), cudaMemcpyHostToDevice);
    }

    public void codeStringNew(NNVector v, boolean TYPE) {
        cudaMemcpy(v.getData_gpu(), Pointer.to(v.getData()), (long) Sizeof.FLOAT * v.size(), cudaMemcpyHostToDevice);
    }

    public String decodeString(float[] input) {
        StringBuilder string = new StringBuilder();
        for (int i = 0; i < input.length; i++) {
            Character Char = (uaChars.get((int) (Math.round((input[i]/* - 1*/) * constant))));
            if (Char != null) {
                string.append(Char);
            }
        }
        return string.toString();
    }

    public String decodeString_new(float[] input) {
        StringBuilder string = new StringBuilder();
        if (WorkingWithCharacter) {
            for (int i = 0; i < input.length; i++) {
                Character Char = (uaChars.get((int) (Math.round((input[i]) * constant))));
                if (Char != null) {
                    string.append(Char);
                }
            }
            /*Use.CPU = true;
            Use.GPU = false;
            NNMatrix arr = new NNMatrix(WordCount, sizeVocabulary, input, null, false);
            NNVector[] words = arr.toVectors();
            for (int i = 0; i < words.length; i++) {
                Character Char = (uaChars.get((int) (Math.round((words[i].indexMaxElement()) * constant))));
                if (Char != null) {
                    string.append(Char);
                }
            }
            Use.GPU = true;
            Use.CPU = false;*/
        }
        else {
            for (int i = 0; i < input.length; i++) {
                String Char = (uaWords.get((int) (Math.round((input[i]) * constant))));
                if (Char != null) {
                    string.append(Char).append(" ");
                }
            }
        }

        return string.toString();
    }

    public String decodeString_TYPE(short[] input) {
        StringBuilder string = new StringBuilder();
        for (int i = 0; i < input.length; i++) {
            Character Char = (uaChars.get((int) (Math.round((bFloat16ToFloat(input[i])/* - 1*/) * constant))));
            if (Char != null) {
                string.append(Char);
            }
        }
        return string.toString();
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

    public static String fileNameWithOutExt(String fileName) {
        return Optional.of(fileName.lastIndexOf(".")).filter(i -> i >= 0)
                .filter(i -> i > fileName.lastIndexOf(File.separator))
                .map(i -> fileName.substring(0, i)).orElse(fileName);
    }
}
