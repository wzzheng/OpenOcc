
import os, os.path as osp
from copy import copy
import torch, time, argparse
import torch.distributed as dist

from model import *
from loss import OPENOCC_LOSS
from utils.build_scheduler import create_scheduler
from utils.average_meter import AverageMeter
from utils.metric_util import MeanIoU
from utils.load_save_util import revise_ckpt, revise_ckpt_2
from utils.dtype_lut import dtypeLut

import mmcv
from mmcv import Config
from mmcv.runner import build_optimizer
from mmseg.utils import get_root_logger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # init DDP
    distributed = True
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = os.environ.get("MASTER_PORT", "20506")
    hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
    rank = int(os.environ.get("RANK", 0))  # node id
    gpus = torch.cuda.device_count()  # gpus per node
    print(f"tcp://{ip}:{port}")
    dist.init_process_group(
        backend="nccl", init_method=f"tcp://{ip}:{port}", 
        world_size=hosts * gpus, rank=rank * gpus + local_rank
    )
    world_size = dist.get_world_size()
    cfg.gpu_ids = range(world_size)
    torch.cuda.set_device(local_rank)

    # disable print from none-0 processes
    if dist.get_rank() != 0:
        import builtins
        builtins.print = pass_print
    
    # dump configuration
    if dist.get_rank() == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))

    # configure logging
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')

    # build model
    my_model = build_segmentor(cfg.model)
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        my_model = my_model.cuda()
    logger.info('done ddp model')

    # generate datasets
    from dataset import get_dataloader
    train_dataset_loader, val_dataset_loader = \
        get_dataloader(
            train_wrapper=cfg.train_wrapper,
            val_wrapper=cfg.val_wrapper,
            train_loader=cfg.train_loader,
            val_loader=cfg.val_loader,
            dist=distributed)
    
    # get metric calculator
    label_str = train_dataset_loader.dataset.loader.nuScenes_label_name
    metric_label = cfg.unique_label
    metric_str = [label_str[x] for x in metric_label]
    metric_ignore_label = cfg.metric_ignore_label
    CalMeanIou_pts = MeanIoU(metric_label, metric_ignore_label, metric_str, 'pts')

    # get optimizer, loss, scheduler
    optimizer = build_optimizer(my_model, cfg.optimizer)
    multi_loss_func = OPENOCC_LOSS.build(cfg.loss)
    cfg.scheduler.update(
        {'num_steps': len(train_dataset_loader) * cfg.max_num_epochs})
    scheduler = create_scheduler(cfg.scheduler, optimizer)
        
    # resume and load
    epoch = 0
    best_val_miou_pts = 0
    global_iter = 0

    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    print('resume from: ', cfg.resume_from)
    print('work dir: ', args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        print(my_model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch = ckpt['epoch']
        if 'best_val_miou_pts' in ckpt:
            best_val_miou_pts = ckpt['best_val_miou_pts']
        global_iter = ckpt['global_iter']
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        state_dict = revise_ckpt(state_dict)
        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:
            print('removing img_neck.lateral_convs and img_neck.fpn_convs')
            state_dict = revise_ckpt_2(state_dict)
            print(my_model.load_state_dict(state_dict, strict=False))
        

    # training
    print_freq = cfg.print_freq
    max_num_epochs = cfg.max_num_epochs
    lossMeter = AverageMeter()

    # eval
    my_model.eval()
    lossMeter.reset()
    CalMeanIou_pts.reset()

    with torch.no_grad():
        for i_iter_val, inputs in enumerate(val_dataset_loader):

            new_inputs = copy(inputs)
            for new_name, old_name, dtype, device in cfg.input_convertion:
                item = inputs[old_name]
                if dtype is not None:
                    item = item.to(dtypeLut[dtype])
                if device is not None:
                    item = item.to(device)
                new_inputs.update({new_name: item})
            data_time_e = time.time()
            
            # forward + backward + optimize
            model_inputs = {}
            for model_arg_name, input_name in cfg.model_inputs.items():
                model_inputs.update({model_arg_name: new_inputs[input_name]})
            model_outputs = my_model(**model_inputs)
            new_inputs.update(model_outputs)
            
            loss_inputs = {}
            for loss_arg_name, input_name in cfg.loss_inputs.items():
                loss_inputs.update({loss_arg_name: new_inputs[input_name]})
            loss, _ = multi_loss_func(loss_inputs)

            predict_labels_pts = new_inputs['outputs_pts']
            val_pt_labs = new_inputs['point_labels']
            predict_labels_pts = predict_labels_pts.squeeze(-1).squeeze(-1)
            predict_labels_pts = torch.argmax(predict_labels_pts, dim=1) # bs, n
            predict_labels_pts = predict_labels_pts.detach().cpu()
            val_pt_labs = val_pt_labs.squeeze(-1).cpu()
            
            for count in range(len(predict_labels_pts)):
                CalMeanIou_pts._after_step(predict_labels_pts[count], val_pt_labs[count])
            
            lossMeter.update(loss.detach().cpu().item())
            if i_iter_val % print_freq == 0 and dist.get_rank() == 0:
                logger.info('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)'%(
                    epoch, i_iter_val, lossMeter.val, lossMeter.avg))
    
    val_miou_pts = CalMeanIou_pts._after_epoch()

    if best_val_miou_pts < val_miou_pts:
        best_val_miou_pts = val_miou_pts
    logger.info('Current val miou pts is %.3f while the best val miou pts is %.3f' %
            (val_miou_pts, best_val_miou_pts))
    logger.info('Current val loss is %.3f' %
            (lossMeter.avg))


    while epoch < max_num_epochs:
        my_model.train()
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        lossMeter.reset()
        data_time_s = time.time()
        time_s = time.time()

        for i_iter, inputs in enumerate(train_dataset_loader):
            
            new_inputs = copy(inputs)
            for new_name, old_name, dtype, device in cfg.input_convertion:
                item = inputs[old_name]
                if dtype is not None:
                    item = item.to(dtypeLut[dtype])
                if device is not None:
                    item = item.to(device)
                new_inputs.update({new_name: item})
            data_time_e = time.time()
            
            # forward + backward + optimize
            model_inputs = {}
            for model_arg_name, input_name in cfg.model_inputs.items():
                model_inputs.update({model_arg_name: new_inputs[input_name]})
            model_outputs = my_model(**model_inputs)
            new_inputs.update(model_outputs)
            
            loss_inputs = {}
            for loss_arg_name, input_name in cfg.loss_inputs.items():
                loss_inputs.update({loss_arg_name: new_inputs[input_name]})
            loss, _ = multi_loss_func(loss_inputs)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
            optimizer.step()
            lossMeter.update(loss.detach().cpu().item())
            scheduler.step_update(global_iter)
            time_e = time.time()

            global_iter += 1
            if i_iter % print_freq == 0 and dist.get_rank() == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), grad_norm: %.1f, lr: %.7f, time: %.3f (%.3f)'%(
                    epoch, i_iter, len(train_dataset_loader), 
                    lossMeter.val, lossMeter.avg, grad_norm, lr,
                    time_e - time_s, data_time_e - data_time_s))
            data_time_s = time.time()
            time_s = time.time()
        
        # save checkpoint
        if dist.get_rank() == 0:
            dict_to_save = {
                'state_dict': my_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
                'best_val_miou_pts': best_val_miou_pts,
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            mmcv.symlink(save_file_name, dst_file)
        
        # eval
        my_model.eval()
        lossMeter.reset()
        CalMeanIou_pts.reset()

        with torch.no_grad():
            for i_iter_val, inputs in enumerate(val_dataset_loader):

                new_inputs = copy(inputs)
                for new_name, old_name, dtype, device in cfg.input_convertion:
                    item = inputs[old_name]
                    if dtype is not None:
                        item = item.to(dtypeLut[dtype])
                    if device is not None:
                        item = item.to(device)
                    new_inputs.update({new_name: item})
                data_time_e = time.time()
                
                # forward + backward + optimize
                model_inputs = {}
                for model_arg_name, input_name in cfg.model_inputs.items():
                    model_inputs.update({model_arg_name: new_inputs[input_name]})
                model_outputs = my_model(**model_inputs)
                new_inputs.update(model_outputs)
                
                loss_inputs = {}
                for loss_arg_name, input_name in cfg.loss_inputs.items():
                    loss_inputs.update({loss_arg_name: new_inputs[input_name]})
                loss, _ = multi_loss_func(loss_inputs)

                predict_labels_pts = new_inputs['outputs_pts']
                val_pt_labs = new_inputs['point_labels']
                predict_labels_pts = predict_labels_pts.squeeze(-1).squeeze(-1)
                predict_labels_pts = torch.argmax(predict_labels_pts, dim=1) # bs, n
                predict_labels_pts = predict_labels_pts.detach().cpu()
                val_pt_labs = val_pt_labs.squeeze(-1).cpu()
                
                for count in range(len(predict_labels_pts)):
                    CalMeanIou_pts._after_step(predict_labels_pts[count], val_pt_labs[count])
                
                lossMeter.update(loss.detach().cpu().item())
                if i_iter_val % print_freq == 0 and dist.get_rank() == 0:
                    logger.info('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)'%(
                        epoch, i_iter_val, lossMeter.val, lossMeter.avg))
        
        val_miou_pts = CalMeanIou_pts._after_epoch()

        if best_val_miou_pts < val_miou_pts:
            best_val_miou_pts = val_miou_pts
        logger.info('Current val miou pts is %.3f while the best val miou pts is %.3f' %
                (val_miou_pts, best_val_miou_pts))
        logger.info('Current val loss is %.3f' %
                (lossMeter.avg))
        
        epoch += 1
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')

    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
