package test.utilitas;

import org.imgscalr.Scalr;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Scanner;

public class TinyImageTest {
    public static void main(String[] args) throws IOException {
        String path = "D:\\datasets\\tiny-imagenet\\val\\images\\";
        String pathOut = "D:\\datasets\\TinyImagenet\\val\\";

        HashMap<String, Integer> label = new HashMap<>();
        Scanner scanner = new Scanner(new File("D:\\datasets\\tiny-imagenet\\label.txt"));
        for (int i = 0; i < 200; i++) {
            label.put(scanner.nextLine(), i);
        }
        int arr[] = new int[200];
        System.out.println(label);

        Scanner scannerVal = new Scanner(new File("D:\\datasets\\tiny-imagenet\\val\\val_annotations.txt"));
        while (scannerVal.hasNextLine()) {
            String[] param = scannerVal.nextLine().split("\t");
            int n = label.get(param[1]);
            BufferedImage imgs = ImageIO.read(new File(path + param[0])); // load image
            try {
                ImageIO.write(imgs, "png", new File(pathOut + (arr[n] * 200 + n) + ".png"));
                arr[n]++;
            } catch (IOException e) {
                e.printStackTrace();
            }

        }

//        for (File file : dir.listFiles()) {
//                n = i;
//                System.out.println(file.getName());
//                for (File img : new File(file.getPath() + "\\images").listFiles()) {
//                    BufferedImage imgs = ImageIO.read(img); // load image
//                    try {
//                        ImageIO.write(imgs, "png", new File(pathOut + n + ".png"));
//                        n += 200;
//                    } catch (IOException e) {
//                        e.printStackTrace();
//                    }
//                }
//                i++;
//        }
    }
}
