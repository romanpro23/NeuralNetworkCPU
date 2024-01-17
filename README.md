**Neural Network CPU
Library deep learning on java.**

This library will provide you with a wide range of possibilities for implementing various neural networks. You can get acquainted with them below:




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

_**Additional resources**_

Find some datasets, that used in this library:
https://drive.google.com/file/d/1pUiCEaB6mgx7vSjgbIQu-WoszNeiM36P/view?usp=sharing

TinyImagenet and imbd:
https://drive.google.com/file/d/1QnbGNTCgi11GyFEXcJY1nidToAOg42kx/view?usp=sharing
