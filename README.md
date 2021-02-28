# Generative Adversarial Networks

### Getting Datasets

##### Building Facades

```shell
wget https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz
tar -xf facades.tar.gz
```

### Running Tensorboard on a remote machine

On your local machine, run:

```
ssh -L 16006:127.0.0.1:6006 [USERNAME]@[IP_ADDRESS]
```

On your remote machine, in the folder of a particular model, run:

```
tensorboard --logdir=runs
```

Go to `http://127.0.0.1:16006` on your local machine
