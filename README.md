# PyTorch GradNorm

<p align="center"><img width="75%" src="images/gradnorm.png" /></p>

This is a PyTorch-based implementation of [GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks](http://proceedings.mlr.press/v80/chen18a.html), which is a gradient normalization algorithm that automatically balances training in deep multitask models by dynamically tuning gradient magnitudes.

The toy example can be found at [**here**](https://github.com/LucasBoTang/GradNorm/blob/main/Test.ipynb).


## Algorithm

<p align="center"><img width="50%" src="images/algo.png" /></p>


## Dependencies

[PyTorch](https://pytorch.org/)

[NumPy](https://numpy.org/)


## Usage

### Parameters

- net: a multitask network with task loss
- layer: a layers of the network layers where appling GradNorm on the weights
- alpha: hyperparameter of restoring force
- dataloader: training dataloader
- num_epochs: number of epochs
- lr1:  learning rate of multitask loss
- lr2:  learning rate of weights
- log:  flag of result log

### Sample Code

```python
log_weights, log_loss = gradNorm(net=net, layer=net.fc4, alpha=0.12, dataloader=dataloader,
                                 num_epochs=100, lr1=1e-5, lr2=1e-4, log=True)
```

