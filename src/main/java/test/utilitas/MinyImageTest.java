package test.utilitas;

import org.imgscalr.Scalr;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.Scanner;

public class MinyImageTest {
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

        Scanner scanner1 = new Scanner(new File("D:\\datasets\\ImageNet100\\labels.txt"));
        Scanner scanner2 = new Scanner(new File("D:\\datasets\\TinyImagenet\\label.txt"));
        String pathOut = "D:\\datasets\\ImageNet\\train\\";

        LinkedHashSet<String> set = new LinkedHashSet<>();

        for (int i = 0; i < 100; i++) {
            set.add(scanner1.nextLine());
        }

        for (int i = 0; i < 200; i++) {
            set.add(scanner2.nextLine());
        }

        System.out.println(set.size());

        FileWriter writer = new FileWriter("D:\\datasets\\ImageNet\\labels.txt");
        File dir = new File("D:\\datasets\\ImageNet\\train");
        int n = 0, i = 0;
        for (File file : dir.listFiles()) {
//            File f = new File(pathOut + file.getName());
//            if (f.listFiles() != null) {
//                n = f.listFiles().length;
//            } else {
//                Files.createDirectories(Path.of(f.getPath()));
//                n = 0;
//            }
            System.out.println(file.getName());
                writer.write(file.getName() + "\n");
                writer.flush();
//            for (File img : file.listFiles()) {
//                BufferedImage imgs = ImageIO.read(img); // load image
//                int h = imgs.getHeight(), w = imgs.getWidth();
//                int s = Math.min(h, w);
//                int hP = (h - s) / 2;
//                int wP = (w - s) / 2;
//                imgs = imgs.getSubimage(wP, hP, s, s);
//                imgs = Scalr.resize(imgs, 64, 64);
//                try {
//                    ImageIO.write(imgs, "png", new File(pathOut + file.getName() + "\\" + n + ".png"));
//                    n++;
//                } catch (IOException e) {
//                    e.printStackTrace();
//                }
//            }
//            i++;
        }
    }
}
//
//package test.utilitas;
//
//        import org.imgscalr.Scalr;
//
//        import javax.imageio.ImageIO;
//        import java.awt.image.BufferedImage;
//        import java.io.File;
//        import java.io.FileWriter;
//        import java.io.IOException;
//        import java.nio.file.Files;
//        import java.nio.file.Path;
//        import java.util.LinkedHashSet;
//        import java.util.Scanner;
//
//public class MinyImageTest {
//    public static void main(String[] args) throws IOException {
//        String pathOut = "D:\\datasets\\ImageNet\\train\\";
//        String pathOutVal = "D:\\datasets\\ImageNet\\valid\\";
//
//        File dir = new File("D:\\datasets\\MinyImagenet\\train");
//        int n, nV = 0, i = 0;
//        for (File file : dir.listFiles()) {
//            File f = new File(pathOut + file.getName());
//            File fV = new File(pathOutVal + file.getName());
//            if (f.listFiles() != null) {
//                n = f.listFiles().length;
//            } else {
//                Files.createDirectories(Path.of(f.getPath()));
//                n = 0;
//            }
//            if (fV.listFiles() != null) {
//                nV = fV.listFiles().length;
//            } else {
//                Files.createDirectories(Path.of(fV.getPath()));
//                nV = 0;
//            }
//            i = 0;
//            System.out.println(file.getName());
//            for (File img : file.listFiles()) {
//                try {
//                    BufferedImage imgs = ImageIO.read(img); // load image
//                    int h = imgs.getHeight(), w = imgs.getWidth();
//                    int s = Math.min(h, w);
//                    int hP = (h - s) / 2;
//                    int wP = (w - s) / 2;
//                    imgs = imgs.getSubimage(wP, hP, s, s);
//                    imgs = Scalr.resize(imgs, 64, 64);
//                    if(i < 50){
//                        ImageIO.write(imgs, "png", new File(pathOutVal + file.getName() + "\\" + nV + ".png"));
//                        nV++;
//                        i++;
//                    } else {
//                        ImageIO.write(imgs, "png", new File(pathOut + file.getName() + "\\" + n + ".png"));
//                        n++;
//                    }
//
//                }  catch (Exception e) {
//                    e.printStackTrace();
//                }
//            }
//        }
//    }
//}