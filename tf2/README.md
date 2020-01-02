# Minimal Product-Key Memory Layer MNIST Example - Tensorflow2

A minimal repo to show training with the [Product-Key Memory Layer](https://arxiv.org/abs/1907.05242)

based on the implementation provided at https://github.com/facebookresearch/XLM/blob/master/PKM-layer.ipynb (credit to facebook)

takes the [Pytorch MNIST Example](https://github.com/pytorch/examples/blob/master/mnist/main.py) and inserts a residual layer between the two fully connected layers.  Depending on activating the layer the layer will be either be a PKM layer or a linear layer

This is mostly a straight conversion to tensorflow 2
- This version is much more memory hungry than the pytorch version, partially because of pytorch's embedding bag
- The optimizer for the PKM layer is the same as for the rest of the network, this is different than the pytorch version,
when I get the chance I will check to see if there is a way to use 2 optimizers

I did this mainly to get some experience in TF2, it's very possible this is not an efficient implementation
I cannot seem to get it to run on GPU on my RTX 2080, Which I think is from memory usage
CPU is pretty slow as always

If you see anything in the code I can fix to be more "proper" for tf2 please open an issue and I'll fix it.


# installation and running

The easiest installation is with using anaconda

## install requirements
```
conda create -yn pkmdemo-tf2 python=3
conda env update -f requirements.yml -n pkmdemo-tf2
```

## Run code

```
conda activate pkmdemo-tf2
python main.py
```

## results


### pkm on - 4 min per epoch on cpu

```
python main.py
Epoch 10, Loss: 0.0006459007854573429, Accuracy: 99.9816665649414, Test Loss: 0.07127846777439117, Test Accuracy: 99.01000213623047
```

### pkm off - less than 10 sec per epoch on RTX 2080

```
python main.py --pkm_off
Epoch 9, Loss: 0.004304786212742329, Accuracy: 99.88666534423828, Test Loss: 0.07075853645801544, Test Accuracy: 99.08000183105469

```