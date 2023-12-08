package test.speechtotext;

import data.loaders.*;
import data.network_train.NNData2D;
import data.nlp.NLP;
import jcuda.Pointer;
import jcuda.Sizeof;
import nnarrays.NNMatrix;
import nnarrays.NNTensor;
import nnarrays.NNVector;
import utilities.Use;

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

public class PositionLoader extends DataLoader2D {
    private LinkedHashMap<Integer, Character> uaChars;
    private LinkedHashMap<Character, Integer> codeUaChars;

    private int sizeBuffer = 5000;

    float constant = 1.0f;

    public PositionLoader(int countChars) throws Exception {
        uaChars = new LinkedHashMap<>();
        codeUaChars = new LinkedHashMap<>();
        train = new ArrayList<>();
        test = new ArrayList<>();

        try {
            Scanner scanner = new Scanner(new File("C:/Levani/NeuralNetworkCPU/data/ka_chars.txt"), StandardCharsets.UTF_8);
            uaChars.put(NLP.PAD, ' ');
            codeUaChars.put(' ', NLP.PAD);
            uaChars.put(NLP.Empty, '_');
            codeUaChars.put('_', NLP.Empty);

            int key = 1;
            for (int i = 0; i < countChars; i++) {
                Character str = scanner.nextLine().charAt(0);
                uaChars.put(key, str);
                codeUaChars.put(str, key);
                key += 1;
            }
        } catch (IOException e) {
            e.printStackTrace();
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

                NNMatrix inputsData = new NNMatrix(24, size_width, true);

                Use.GPU = false;

                for (int xx = 0; xx < width; xx++) {
                    for (int yy = 0; yy < height; yy++) {
                        Color color = new Color(img.getRGB(xx, yy));
                        inputsData.set(yy, width - xx - 1, transformData.transformR(color.getRed()));
                    }
                }

                /*for (int xx = 0; xx < width; xx++) {
                    for (int yy = 0; yy < height; yy++) {
                        Color color = new Color(img.getRGB(xx, yy));
                        inputsData.set(xx, yy, transformData.transformR(color.getRed()));
                    }
                }*/

                Use.GPU = true;

                StringBuilder label = new StringBuilder();
                for (String[] datum : data) {
                    if (Objects.equals(fileNameWithOutExt(datum[1]), fileNameWithOutExt(afile.getName()))) {
                        label = new StringBuilder(datum[2]);
                        break;
                    }
                }

                if (!Objects.equals(label.toString(), "")) {
                    int c = 176;
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
                    }
                    for (int ww = 0; ww < sss; ww++) {
                        label.append("_");
                    }

                    NNMatrix inputsDataNew = new NNMatrix(inputsData.getRow(), inputsData.getColumn(), inputsData.getData(), inputsData.getSdata(), true);

                    inputsDataNew.ClearCpuData();
                    inputsData = null;
                    inputsDataNew.ClearCpuData();

                    Use.CPU = false;

                    NNVector output = codeString(label.toString(), true);

                    train.add(new ImageData2D(inputsDataNew, output));
                    test.add(new ImageData2D(inputsDataNew, output));

                    if (wwq % 100 == 0) {
                        System.out.println(wwq);
                    }

                    //if (wwq == 99) {
                    //  return;
                    //}

                    wwq++;
                }

                Use.CPU = false;
            }
        }
    }

    public NNVector codeString(String text, boolean TYPE) {
        char[] chars = text.toCharArray();
        NNVector input = new NNVector(chars.length, TYPE);
        if (Use.CPU) {
            for (int j = 0; j < input.size(); j++) {
                float value = ((float) codeUaChars.get(chars[j]) / constant/* + 1*/);
                input.set(j, value);
            }
        }

        if (Use.GPU) {
            if (!TYPE) {
                float[] sm = new float[input.size()];
                for (int j = 0; j < input.size(); j++) {
                    try {
                        sm[j] = ((float) codeUaChars.get(chars[j]) / constant/* + 1*/);
                    } catch (Exception e) {
                        int ss = 0;
                    }
                }

                cudaMemcpy(input.getData_gpu(), Pointer.to(sm), (long) Sizeof.FLOAT * input.size(), cudaMemcpyHostToDevice);
            }
            else
            {
                short[] sm = new short[input.size()];
                for (int j = 0; j < input.size(); j++) {
                    float value = 0;

                    try {
                        value = ((float) codeUaChars.get(chars[j]) / constant/* + 1*/);
                        sm[j] = floatToBFloat16(value);
                    } catch (Exception e) {
                        int ss = 0;
                    }
                }

                cudaMemcpy(input.getData_gpu(), Pointer.to(sm), (long) Sizeof.SHORT * input.size(), cudaMemcpyHostToDevice);
            }
        }

        return input;
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
