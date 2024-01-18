**Neural Network CPU
Library deep learning on java.**

This library will provide you with a wide range of possibilities for implementing various neural networks. You can get acquainted with them below:

_**Mathematical calculations**_

This library contains a package that is responsible for the optimized implementation of arrays and mathematical operations on them. In it, you can see implemented arrays of various dimensions, from vectors to 4-dimensional tensors. In addition to standard operations such as addition, subtraction, multiplication, division, you can perform many other specialized operations on these arrays, such as convolution, matrix multiplication, damping, which is used in optimizers, and much more. Below is an example of the work:

```
NNMatrix a = new NNMatrix(4, 6);
NNMatrix b = new NNMatrix(6, 6);
new Initializer.RandomUniform().initialize(a);
new Initializer.RandomUniform().initialize(b);

System.out.println(a.dot(b));
System.out.println(a.dot(b.transpose()));
```

_**Layers and creating neural networks from them**_

This library provides an intuitive interface for working with neural layers and designing various architectures. Let's take a look at creating a simple neural network

```
NeuralNetwork network = new NeuralNetwork()
                .addInputLayer(784)
                .addLayer(new DenseLayer(64).setTrainable(true))
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DenseLayer(64).setTrainable(true))
                .addActivationLayer(new FunctionActivation.ReLU())
                .addLayer(new DenseLayer(10).setTrainable(false))
                .addLayer(new ActivationLayer(new FunctionActivation.Softmax()))
                .create();
```

The library implements a variety of layers, here is an incomplete list:
- fully connected layers;
- convolutional layers (1-dimensional, 2-D, 3-D);
- modifications of convolutional layers (group convolution, dilatation convolution, transposed convolution, deformed convolution);
- various activation layers with many functions implemented;
- normalization layers, including spectral normalization used for GAN;
- recurrent layers (Simple, GRU, LSTM);
- recurrent layers with attention;
- regularization layers;
- pooling layers;
- layers for implementing different attention mechanisms;
- layers of capsule neural networks.

Thanks to the user-friendly interface, you can quickly and easily create your own layers, supplementing and developing the library.

_**Model settings - optimizers, loss functions, regularizers, and more**_

The library implements a lot of different elements that are needed to train neural networks.

For example, there is a large number of different _optimizers_ that are not available in any other machine learning library, and you can choose any of them for your needs. Or, if necessary, you can easily create your own by simply inheriting from an optimizer class. 
```
public class MomentumOptimizer extends Optimizer {
    /**
     * rt - retention rate
     * Momentum
     * θ(t) = rt * θ(t-1) + (1 - rt) * dw(t)
     * w(t) = w(t-1) - lr * θ(t)
     */
    private final float learningRate;
    private final float retentionRate;

    public MomentumOptimizer(double learningRate, double retentionRate) {
        this.learningRate = (float) learningRate;
        this.retentionRate = (float) retentionRate;
        this.countParam = 1;
    }

    @Override
    public void updateWeight(NNArray weight, NNArray deltaWeight, NNArray[] additionParam) {
        additionParam[0].momentum(deltaWeight, retentionRate);
        weight.subAndMul(additionParam[0], learningRate);
        deltaWeight.clear();
    }
}
```

Below you can see a list of currently available optimizers:

![image](https://github.com/romanpro23/NeuralNetworkCPU/assets/87851373/192aeb3c-a7b7-40fa-b3b7-fdaf5eb65671)

If you need to use some standard _loss function_, why write it from 0 if everything you need is already implemented in the library?

Also, to get started, we need to _initialize_ the weights of the neural layers with some values, and you can use any of the available initializers for this. The library contains all known initializers, and you can easily use them in your development.

If you have problems with retraining, try using _regularizers_ in your neural layers. Regularizers are implemented in the library, and thanks to the well-thought-out structure of layers, you can add a regularizer to a layer immediately when creating it.

If you need to set the parameters for a neural network right when you create it, for example, you are building a classification model, here's how to do it:

```
        NeuralNetwork vgg = NeuralNetwork
                .read(new Scanner(new File("/Imagenet/vgg16.txt")))
                .setFunctionLoss(new FunctionLoss.MSE())
                .setOptimizer(new AdamOptimizer())
                .create();
```

_**Implemented architectures**_

If you don't want to spend your time building popular models, you can use ready-made solutions, saving a lot of your own time. It is done quite simply and quickly:

```
        NeuralNetwork mobilenetV1 = new MobileNet()
                .createMobileNetV1(64,64, 3, 0.5, 200,0.2)
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .setOptimizer(optimizer)
                .create();
```

Or if you want to model your architecture based on an existing one, then use the following mechanisms available in code:

```
        NeuralNetwork inceptionV3 = new Inception()
                .addInputLayer(64, 64, 3)
                .addConvolutionLayer(16, 3, 2,1)
                .addConvolutionLayer(16, 3)
                .addConvolutionLayer(32, 3)
                .addMaxPoolingLayer(3, 2)
                .addConvolutionLayer(40, 1)
                .addConvolutionLayer(96, 3)
                .addMaxPoolingLayer(3, 2)
                .addInceptionA(16, 2)
                .addInceptionA(32, 2)
                .addInceptionA(32, 2)
                .addInceptionB()
                .addInceptionC(5, 128/2, 2)
                .addInceptionC(5, 160/2, 2)
                .addInceptionC(5, 160/2, 2)
                .addInceptionC(5, 192/2, 2)
                .addInceptionD(5, 2)
                .addInceptionE(2)
                .addInceptionE(2)
                .addGlobalAveragePoolingLayer()
                .addDropoutLayer(0.4)
                .addDenseLayer(200, new FunctionActivation.Softmax())
                .createInception()
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
                .setOptimizer(new AdamOptimizer())
                .create();
```

In addition, in the library you will find other architectures, _autoencoders, GAN, recurrent models with attention_ for working with text, _transformers, capsule neural networks_ and others.

_Serialization and deserialization_ in a human-readable format is also implemented. You can train a neural network, save it to a file, and then use it wherever you need it.

_**Ready sets of data**_

Since training any network requires data, the library has collected the most popular datasets in the data package, which are immediately available in a format read by neural networks.
You can create your own datasets simply by inheriting from the Loader class and writing the necessary logic.
```
public abstract class DataLoader {
    public abstract NNData getNextTrainData(int sizeBatch);

    public abstract NNData getNextTestData(int sizeBatch);
}
```
Links to the data are at the very end of this tutorial.

_**Additional resources**_

Find some datasets, that used in this library:
https://drive.google.com/file/d/1pUiCEaB6mgx7vSjgbIQu-WoszNeiM36P/view?usp=sharing

TinyImagenet and imbd:
https://drive.google.com/file/d/1QnbGNTCgi11GyFEXcJY1nidToAOg42kx/view?usp=sharing
