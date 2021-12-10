import argparse
import datetime
import torch
from vit_pytorch import ViT, MAE
from vit_pytorch.ema import EMAHelper
from datasets import get_dataset, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from vit_pytorch.warmup_scheduler import WarmUpLr, GradualWarmupScheduler
import utils
from torchvision.utils import save_image
import torch.utils.tensorboard as tb
import time
import shutil
import numpy as np
import logging
import yaml
import os
import sys
import copy


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, required=True,  help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, required=True, help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--datasets_path', type=str, default='', help='Path for datasets.')
    
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, 'logs', args.doc)

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def model_fn(config):
    v = ViT(
        image_size = config.model.image_size,
        patch_size = config.model.patch_size,
        num_classes = config.model.vit_num_classes,
        dim = config.model.vit_dim,
        depth = config.model.vit_depth,
        heads = config.model.vit_heads,
        mlp_dim = config.model.vit_mlp_dim
    )
    mae = MAE(
        encoder = v,
        masking_ratio = config.model.mask_ratio,   # the paper recommended 75% masked patches
        decoder_dim = config.model.decoder_width,  # paper showed good results with just 512
        decoder_depth = config.model.decoder_depth # anywhere from 1 to 8
    )
    return mae


def optim_fn(config, parameters):
    if config.optim.optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'AdamW':
        return torch.optim.AdamW(parameters, lr=config.optim.warmup_lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return torch.optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return torch.optim.SGD(parameters, lr=config.optim.lr, momentum=config.optim.momentum)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))


def build_model(config):
    model = model_fn(config)

    return model


def build_optimizer(config, model):
    opt = optim_fn(config, model.parameters())

    return opt


'''
 dummy dataloader for test
'''
def sample_unlabelled_images():
    return torch.FloatTensor(128, 3, 256, 256).uniform_(0., 1.)


def train():
    # args and config
    args, config = parse_args_and_config()
    utils.init_distributed_mode(config)

    # logging
    if config.training.gpu == 0:
        if not args.test:
            if not args.resume_training:
                if os.path.exists(args.log_path):
                    overwrite = False
                    # if args.ni:
                    #     overwrite = True
                    # else:
                    #     response = input("Folder already exists. Overwrite? (Y/N)")
                    #     if response.upper() == 'Y':
                    #         overwrite = True
                    overwrite = True
                    
                    if overwrite:
                        shutil.rmtree(args.log_path)
                        os.makedirs(args.log_path)
                        os.makedirs(os.path.join(args.log_path, 'rec_and_gt'))
                    else:
                        print("Folder exists. Program halted.")
                        sys.exit(0)
                else:
                    os.makedirs(args.log_path)
                    os.makedirs(os.path.join(args.log_path, 'rec_and_gt'))

        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

        logging.info("Writing log file to {}".format(args.log_path))
        logging.info("Exp instance id = {}".format(os.getpid()))
        logging.info("Exp comment = {}".format(args.comment))
        logging.info("Config =")
        logging.info(">" * 80)
        config_dict = copy.copy(vars(config))
        logging.info(yaml.dump(config_dict, default_flow_style=False))
        logging.info("<" * 80)

    # seed and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # model
    model = build_model(config)
    patch_size = config.model.patch_size
    logging.info("Patch size = %s" % str(patch_size))
    args.window_size = (config.model.image_size // patch_size, config.model.image_size // patch_size)
    args.patch_size = patch_size
    args.mask_ratio = config.model.mask_ratio
    
    # dataset
    dataset_train, dataset_test = get_dataset(args, config)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank
    num_training_steps_per_epoch = len(dataset_train) // config.training.batch_size // num_tasks
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_mem,
        drop_last=True,
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=config.training.batch_size, 
        shuffle=True,
        num_workers=config.data.num_workers, 
        drop_last=True
    )

    test_iter = iter(dataloader_test)

    # model for distributed training
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info("Model = %s" % str(model_without_ddp))
    logging.info('number of params: {} M'.format(n_parameters / 1e6))

    total_batch_size = config.training.batch_size * utils.get_world_size()
    config.optim.lr = config.optim.lr * total_batch_size / 256

    logging.info("LR = %.8f" % config.optim.lr)
    logging.info("Batch size = %d" % total_batch_size)
    logging.info("Number of training steps = %d" % num_training_steps_per_epoch)
    logging.info("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if config.training.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.training.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # optimizer and learning rate scheduler
    optimizer = build_optimizer(config, model)
    # lr_scheduler = WarmUpLr(optimizer, [100, 400, 1000], warmup_iters=config.training.warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        config.optim.T_0, T_mult=config.optim.T_mul, 
        eta_min=config.optim.min_lr, last_epoch=-1)
    lr_scheduler = GradualWarmupScheduler(
        optimizer, 
        multiplier=config.optim.multiplier, 
        total_epoch=config.optim.warmup_epochs, 
        after_scheduler=cosine_scheduler)

    start_epoch = 0
    step = 0

    # ema
    '''
    ToDo 2021-12-06: may be deleted in the future.
    '''
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(model)

    if args.resume_training:
        states = torch.load(os.path.join(args.log_path, 'checkpoint.pth'))
        score.load_state_dict(states[0])
        ### Make sure we can resume with different eps
        states[1]['param_groups'][0]['eps'] = config.optim.eps
        optimizer.load_state_dict(states[1])
        start_epoch = states[2]
        step = states[3]
        if config.model.ema:
            ema_helper.load_state_dict(states[4])

    # save config
    if config.training.gpu == 0: # main thread save config, logging in stdout
        with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    # training procedure
    start_time = time.time()
    for epoch in range(start_epoch, config.training.n_epochs):
        if config.training.distributed:
            dataloader_train.sampler.set_epoch(epoch)

        for i, (batch, _) in enumerate(dataloader_train):
            step += 1
            images = batch.to(device, non_blocking=True)
            loss, pred, gt = model(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            cur_lr = optimizer.param_groups[0]['lr']
            
            if step % config.training.train_verbose_freq == 0:
                eta = time.time() - start_time
                logging.info("epoch: {}, step: {}, train_loss: {}, lr: {}, eta: {}".format(epoch, step, loss.item(), cur_lr, eta))
                # print(pred.size(), gt.size())
                pred_img = utils.to_image(pred.cpu().data, args)
                gt_img = utils.to_image(gt.cpu().data, args)
                save_image(pred_img, os.path.join(args.log_path, 'train_pred_img_{}_{}_{}.png'.format(epoch,step,args.local_rank)))
                save_image(gt_img, os.path.join(args.log_path, 'train_gt_img_{}_{}_{}.png'.format(epoch,step,args.local_rank)))

            if step % config.training.evaluate_freq == 0:
                if config.model.ema:
                    test_model = ema_helper.ema_copy(model)
                else:
                    test_model = model
                
                test_model.eval()
                try:
                    test_X, test_y = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    test_X, test_y = next(test_iter)

                test_X = test_X.to(device)

                with torch.no_grad():
                    test_loss, pred, gt = test_model(test_X)
                    eta = time.time() - start_time
                    logging.info("epoch: {}, step: {}, test_loss: {}, eta: {}".format(epoch, step, test_loss.item(), eta))
                    pred_img = utils.to_image(pred.cpu().data, args)
                    gt_img = utils.to_image(gt.cpu().data, args)
                    save_image(pred_img, os.path.join(args.log_path, 'test_pred_img_{}_{}_{}.png'.format(epoch,step,args.local_rank)))
                    save_image(gt_img, os.path.join(args.log_path, 'test_gt_img_{}_{}_{}.png'.format(epoch,step,args.local_rank)))

                    del test_model

            if step % config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    epoch,
                    step,
                ]
                if config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(states, os.path.join(args.log_path, 'checkpoint_{}.pth'.format(step)))
                torch.save(states, os.path.join(args.log_path, 'checkpoint.pth'))
        
        lr_scheduler.step()



    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    train()