# mae-repo
PyTorch re-implememtation of "masked autoencoders are scalable vision learners".
In this repo, it heavily borrows codes from codebase https://github.com/lucidrains/vit-pytorch (for MAE architectures) and https://github.com/pengzhiliang/MAE-pytorch (for training loop).

## prepare ImageNet1K datasets
To train MAE, one should prepare ImageNet_ILSVRC2012 and place ILSVRC2012_*.tar in the ${datasets_path}. To shorten the overhead of first run, one can manually untar the tarfile into train and val directories, as follow (refered to https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
```

```
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```
## modify configuration file
To separate code and config, we try to split configurations to yaml file, located in configs directory, such as imagenet1k-vit-base.yml.
One can modify 'model' setting following [MAE](https://arxiv.org/abs/2111.06377) and [ViT](https://arxiv.org/abs/2010.11929) to configure model architecture parameters of ViT-base, large and huge.

One can modify 'optim' for optimizer settings. And modify 'training' and 'data' for training settings. Note that, modify 'training:batch_size' to fit the GPU memory of one GPU card. Total batch_size is equal to batch_size multiplied by number of GPU cards.

## train
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 mae_test.py \
        --datasets_path ${datasets_path} \
        --config imagenet1k-vit-base.yml \
        --doc mae-vit-base16-dec8-512
        
## ToDo lists
- [x] add pretrain mode
- [ ] add fine-tunning mode
- [ ] support mixed precision training
- [x] support distributed training
- [ ] verify the correctness of this re-implementation
