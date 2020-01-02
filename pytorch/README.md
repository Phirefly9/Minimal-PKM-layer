# Minimal Product-Key Memory Layer MNIST Example - Pytorch

A minimal repo to show training with the [Product-Key Memory Layer](https://arxiv.org/abs/1907.05242)

based on the implementation provided at https://github.com/facebookresearch/XLM/blob/master/PKM-layer.ipynb (credit to facebook)

takes the [Pytorch MNIST Example](https://github.com/pytorch/examples/blob/master/mnist/main.py) and inserts a residual layer between the two fully connected layers.  Depending on activating the layer the layer will be either be a PKM layer or a linear layer

# installation and running

The easiest installation is with using anaconda

## install requirements
```
conda create -yn pkmdemo python=3
conda env update -f requirements.yml -n pkmdemo
```

## Run code

```
conda activate pkmdemo
python main.py
```

## results


### pkm on

```
python main.py
Test set: Average loss: 0.0395, Accuracy: 9923/10000 (99%)
```

### pkm off

```
python main.py --pkm_off
Test set: Average loss: 0.0512, Accuracy: 9910/10000 (99%)
```