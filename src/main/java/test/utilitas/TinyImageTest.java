package test.utilitas;

import org.imgscalr.Scalr;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class TinyImageTest {
    public static void main(String[] args) throws IOException {
//        String path = "D:\\datasets\\tiny-imagenet\\val\\images\\";
//        String pathOut = "D:\\datasets\\TinyImagenet\\val\\";
//
//        HashMap<String, Integer> label = new HashMap<>();
//        Scanner scanner = new Scanner(new File("D:\\datasets\\tiny-imagenet\\label.txt"));
//        for (int i = 0; i < 200; i++) {
//            label.put(scanner.nextLine(), i);
//        }
//        int arr[] = new int[200];
//        System.out.println(label);
//
//        Scanner scannerVal = new Scanner(new File("D:\\datasets\\tiny-imagenet\\val\\val_annotations.txt"));
//        while (scannerVal.hasNextLine()) {
//            String[] param = scannerVal.nextLine().split("\t");
//            int n = label.get(param[1]);
//            BufferedImage imgs = ImageIO.read(new File(path + param[0])); // load image
//            try {
//                ImageIO.write(imgs, "png", new File(pathOut + (arr[n] * 200 + n) + ".png"));
//                arr[n]++;
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//
//        }

//        Scanner scanner1 = new Scanner(new File("D:\\datasets\\ImageNet100\\labels.txt"));
//        Scanner scanner2 = new Scanner(new File("D:\\datasets\\TinyImagenet\\label.txt"));
        String pathOut = "D:\\datasets\\ImageNet\\train\\";
//
//        LinkedHashSet<String> set = new LinkedHashSet<>();
//
//        for (int i = 0; i < 100; i++) {
//            set.add(scanner1.nextLine());
//        }
//
//        for (int i = 0; i < 200; i++) {
//            set.add(scanner2.nextLine());
//        }

//        System.out.println(set.size());

//        FileWriter writer = new FileWriter("D:\\datasets\\ImageNet100\\labels.txt");
        File dir = new File("D:\\datasets\\ImageNet100\\train");
        int n = 0, i = 0;
        for (File file : dir.listFiles()) {
            File f = new File(pathOut + file.getName());
            Files.createDirectories(Path.of(f.getPath()));
            n = 0;
            System.out.println(file.getName());
            for (File img : file.listFiles()) {
                try {
                    BufferedImage imgs = ImageIO.read(img); // load image
                    imgs = Scalr.resize(imgs, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.FIT_EXACT, 64, 64);
                    ImageIO.write(imgs, "png", new File(pathOut + file.getName() + "\\" + n + ".png"));
                    n++;
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
            i++;
        }
    }
}
