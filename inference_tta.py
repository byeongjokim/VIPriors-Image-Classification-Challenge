#!/usr/bin/env python
"""PyTorch Inference Script

An example inference script that outputs top-k class ids for images in a folder into a csv.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import logging
import numpy as np
import torch

from timm.models import create_model, apply_test_time_pool
from timm.data import Dataset, create_loader, resolve_data_config
from timm.utils import AverageMeter, setup_default_logging
from timm.utils import *

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output_dir', metavar='DIR', default='./',
                    help='path to output files')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number classes in dataset')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--topk', default=5, type=int,
                    metavar='N', help='Top-k to output to CSV')
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

parser.add_argument('--outcsv', type=str, default="output")

parser.add_argument('--tta', type=int, default=3)

parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')

def main():
    torch.cuda.set_device(1)
    setup_default_logging()
    args = parser.parse_args()
    # might as well try to do something useful...
    args.pretrained = args.pretrained or not args.checkpoint

    # create model
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=3,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint)

    logging.info('Model %s created, param count: %d' %
                 (args.model, sum([m.numel() for m in model.parameters()])))

    config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = apply_test_time_pool(model, config, args)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()

    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.checkpoint)    

        model = model_ema.ema    

    loader = create_loader(
        Dataset(args.data),
        input_size=config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,

        is_training=True,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,

        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.workers,
        crop_pct=1.0 if test_time_pool else config['crop_pct'])

    model.eval()

    k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()
    topk_ids = []
    scores = []

    predicts = []
    with torch.no_grad():
        for c in range(args.tta):
            predict = [] 
            for batch_idx, (input, _) in enumerate(loader):
                input = input.cuda()
                labels = model(input)
                labels = torch.nn.functional.softmax(labels)
                
                predict.append(labels.cpu().numpy())
            
                if batch_idx % args.log_freq == 0:
                    logging.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                        batch_idx, len(loader), batch_time=batch_time))
            
            predict = np.concatenate(predict, axis=0).squeeze()
            predicts.append(predict)
            
        
        predicts = np.array(predicts)
        predicts = predicts.mean(axis=0)
        
        idx = (-predicts).argsort()[:, :args.topk]
        
        results = []
        for i in range(len(idx)):
            results.append(predicts[i, idx[i]])
        results = np.array(results)
        
    import pandas
    filenames = loader.dataset.filenames()    
    df = pandas.DataFrame(data={"Image IDs": [os.path.basename(f) for f in filenames], "Predictions1": idx[:, 0], "Predictions2":idx[:, 1], "Predictions3":idx[:, 2], "Predictions4":idx[:, 3], "Predictions5":idx[:, 4], "Score1":results[:,0], "Score2":results[:,1], "Score3":results[:,2], "Score4":results[:,3], "Score5":results[:,4]})
    df.to_csv("./"+args.outcsv+"_tta.csv", sep=',', index=False)

if __name__ == '__main__':
    main()
