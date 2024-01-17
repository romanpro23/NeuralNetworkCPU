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
                .setOptimizer(new AdamOptimizer())
                .setFunctionLoss(new FunctionLoss.CrossEntropy())
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

For example, there is a large number of different optimizers that are not available in any other machine learning library, and you can choose any of them for your needs. Or, if necessary, you can easily create your own by simply inheriting from an optimizer class. 
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

![image](https://github.com/romanpro23/NeuralNetworkCPU/assets/87851373/2064fdb4-3782-4c1c-90b2-85e80ac4f14b)



_**Additional resources**_

Find some datasets, that used in this library:
https://drive.google.com/file/d/1pUiCEaB6mgx7vSjgbIQu-WoszNeiM36P/view?usp=sharing

TinyImagenet and imbd:
https://drive.google.com/file/d/1QnbGNTCgi11GyFEXcJY1nidToAOg42kx/view?usp=sharing
