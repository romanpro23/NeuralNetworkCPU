package data.mnist;

import lombok.Getter;

public enum BatchMNIST {
    MNIST("mnist/train-images.idx3-ubyte",
            "mnist/train-labels.idx1-ubyte",
            "mnist/t10k-images.idx3-ubyte",
            "mnist/t10k-labels.idx1-ubyte",
            10,
            60000,
            10000),
    FASHION_MNIST("fashion_mnist/train-images-idx3-ubyte",
            "fashion_mnist/train-labels-idx1-ubyte",
            "fashion_mnist/t10k-images-idx3-ubyte",
            "fashion_mnist/t10k-labels-idx1-ubyte",
            10,
            60000,
            10000),
    EMNIST_BALANCED("emnist_balanced/emnist-balanced-train-images-idx3-ubyte",
            "emnist_balanced/emnist-balanced-train-labels-idx1-ubyte",
            "emnist_balanced/emnist-balanced-test-images-idx3-ubyte",
            "emnist_balanced/emnist-balanced-test-labels-idx1-ubyte",
            47,
            112800,
            18800);

    private String trainFile;
    private String trainFileMark;
    private String testFile;
    private String testFileMark;
    @Getter
    private int sizeTrain;
    @Getter
    private int sizeTest;

    private int countClass;

    BatchMNIST(String trainFile, String trainFileMark, String testFile, String testFileMark, int countClass, int sizeTrain, int sizeTest) {
        this.trainFile = trainFile;
        this.trainFileMark = trainFileMark;
        this.testFile = testFile;
        this.testFileMark = testFileMark;
        this.countClass = countClass;

        this.sizeTest = sizeTest;
        this.sizeTrain = sizeTrain;
    }

    public String getTrainFile() {
        return trainFile;
    }

    public String getTrainFileMark() {
        return trainFileMark;
    }

    public String getTestFile() {
        return testFile;
    }

    public String getTestFileMark() {
        return testFileMark;
    }

    public int getCountClass() {
        return countClass;
    }
}
