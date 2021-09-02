import os
import json
import argparse
import math
import numpy as np
import copy
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
import matplotlib.pyplot as plt

from apex.parallel import DistributedDataParallel as DDP
from apex import amp

from data_utils import TextMelLoader, TextMelCollate
import models
import commons
import utils
from text.symbols import symbols
                            

global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."
  torch.cuda.empty_cache()

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '80001'

  hps = utils.get_hparams()
  mp.spawn(train_and_eval, nprocs=n_gpus, args=(n_gpus, hps,))


def train_and_eval(rank, n_gpus, hps):
  global global_step
  
  ## Added as part of MSc Thesis - Transformer optimization
  global global_omega
  global prev_l_head_wt
  global prev_l_qry_wt
  
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextMelLoader(hps.data.training_files, hps.data)
  train_sampler = torch.utils.data.distributed.DistributedSampler(
      train_dataset,
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextMelCollate(1)
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False,
      batch_size=hps.train.batch_size, pin_memory=True,
      drop_last=True, collate_fn=collate_fn, sampler=train_sampler)
  if rank == 0:
    val_dataset = TextMelLoader(hps.data.validation_files, hps.data)
    val_loader = DataLoader(val_dataset, num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=True, collate_fn=collate_fn)

  generator = models.FlowGenerator(
      n_vocab=len(symbols) + getattr(hps.data, "add_blank", False), 
      out_channels=hps.data.n_mel_channels, 
      **hps.model).cuda(rank)
  if hps.model.mask_flag == 'Y':
      dim_m = (hps.model.hidden_channels / hps.model.n_heads) * (hps.model.n_heads - len(hps.model.mask_heads))
  else:
      dim_m = hps.model.hidden_channels
      #print(dim_m)
  optimizer_g = commons.Adam(generator.parameters(), scheduler=hps.train.scheduler, dim_model=dim_m, warmup_steps=hps.train.warmup_steps, lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
  if hps.train.fp16_run:
    generator, optimizer_g._optim = amp.initialize(generator, optimizer_g._optim, opt_level="O1")
  generator = DDP(generator)
  epoch_str = 1
  global_step = 0
  
  ## Added as part of MSc Thesis - Transformer optimization
  global_omega = np.zeros((4, 8), dtype=float)
  prev_l_head_wt = np.zeros((4, 8), dtype=float)
  prev_l_qry_wt = np.zeros((4, 8), dtype=float)
  
  
  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), generator, optimizer_g)
    epoch_str += 1
    optimizer_g.step_num = (epoch_str - 1) * len(train_loader)
    optimizer_g._update_learning_rate()
    global_step = (epoch_str - 1) * len(train_loader)
    global_omega = 0

  except:
    if hps.train.ddi and os.path.isfile(os.path.join(hps.model_dir, "ddi_G.pth")):
      _ = utils.load_checkpoint(os.path.join(hps.model_dir, "ddi_G.pth"), generator, optimizer_g)
  loss_train = []
  loss_val = []
  
  best_epoch = 1
  loss_diff = 0.0
  cnt = 0

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train(rank, epoch, hps, generator, optimizer_g, train_loader, logger, writer,loss_train)
      evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval,loss_val)
      

      utils.save_checkpoint(generator, optimizer_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(epoch)))
    else:
      train(rank, epoch, hps, generator, optimizer_g, train_loader, None, None)
   
  ## Added as part of MSc Thesis - Transformer optimization
  print("Loss: ",loss_train)
  print("Loss Val: ",loss_val)  
#     loss_diff = abs(loss_val[epoch-1].item() - loss_train[epoch-1].item())
# #    print("loss_diff: ",loss_diff)
#     if loss_diff > 0.1:
#         best_epoch = epoch - 1
#         cnt += 1
#         print("cnt: ",cnt)
#     else:
#         cnt = 0
#         best_epoch = epoch - 1
#     if cnt > 5:
#         break
#   print("loss: ",loss_val)
#   print("Best Epoch: ",best_epoch)
  
  fig = plt.figure()
  plt.title("Loss vs. Number of Training Epochs")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.plot(range(1,hps.train.epochs + 1),loss_train,label='Train')
  plt.plot(range(1,hps.train.epochs + 1),loss_val,label='Validation')
#   plt.plot(range(1,best_epoch + 2),loss_train,label='Train')
#   plt.plot(range(1,best_epoch + 2),loss_val,label='Validation')
  plt.legend()
  fignm="/content/gdrive/MyDrive/Colab Notebooks/Project/glow-tts/logs/fig_"+str(epoch)+".png"
  fig.savefig(fignm)
  print("global_omega: ",global_omega)
  print("global_omega_idx_sort: ", np.argsort(global_omega))
  arr1 = np.array(global_omega).flatten()
  print(arr1.argsort())

def train(rank, epoch, hps, generator, optimizer_g, train_loader, logger, writer,loss_train):
  train_loader.sampler.set_epoch(epoch)
  global global_step
  
  ## Added as part of MSc Thesis - Transformer optimization
  global global_omega
  global prev_l_head_wt
  global prev_l_qry_wt
  losses_tot1 = []
  omega = global_omega
  grad = np.zeros((4, 8), dtype=float)
  
  generator.train()
  for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(train_loader):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)


    # Train Generator
    optimizer_g.zero_grad()
    
    (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask, l_head_wt, l_qry_wt, l_attn_wt), (attn, logw, logw_) = generator(x, x_lengths, y, y_lengths, gen=False)
    l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
    l_length = commons.duration_loss(logw, logw_, x_lengths)

    loss_gs = [l_mle, l_length]
    loss_g = sum(loss_gs)
    #print("prev_l_qry_wt: ",prev_l_qry_wt)
    if batch_idx == 0:
        losses_tot1 = loss_gs
    else:
        losses_tot1 = [x + y for (x, y) in zip(losses_tot1, loss_gs)]
    

    ## Added as part of MSc Thesis - Transformer optimization    
    diff_qry = np.abs(l_qry_wt - prev_l_qry_wt)
    #print("diff_qry: ",diff_qry)
    grad = (np.abs(l_head_wt - prev_l_head_wt)/ l_head_wt) * diff_qry 
    current_size = (batch_idx+1)* hps.train.batch_size #batch_size - 8
    step_size = 1/float(current_size)
    
    #Incremental update for the omega
    omega = omega + step_size*grad 

    if hps.train.fp16_run:
      with amp.scale_loss(loss_g, optimizer_g._optim) as scaled_loss:
        scaled_loss.backward()
      grad_norm = commons.clip_grad_value_(amp.master_params(optimizer_g._optim), 5)
    else:
      loss_g.backward()
      grad_norm = commons.clip_grad_value_(generator.parameters(), 5)
    optimizer_g.step()
    
    if rank==0:
      if batch_idx % hps.train.log_interval == 0:
        (y_gen, *_), *_ = generator.module(x[:1], x_lengths[:1], gen=True)
        logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(x), len(train_loader.dataset),
          100. * batch_idx / len(train_loader),
          loss_gs[0].item()))
        logger.info([x.item() for x in loss_gs] + [global_step, optimizer_g.get_lr()])
        
        scalar_dict = {"loss/g/total": loss_g, "learning_rate": optimizer_g.get_lr(), "grad_norm": grad_norm}
        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(loss_gs)})
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images={"y_org": utils.plot_spectrogram_to_numpy(y[0].data.cpu().numpy()), 
            "y_gen": utils.plot_spectrogram_to_numpy(y_gen[0].data.cpu().numpy()), 
            "attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy()),
            },
          scalars=scalar_dict)
    global_step += 1
    
    ## Added as part of MSc Thesis - Transformer optimization
    prev_l_head_wt = copy.deepcopy(l_head_wt)
    prev_l_qry_wt = copy.deepcopy(l_qry_wt)
    #print("global_step: ",global_step)
    global_omega += (1/((hps.train.epochs + 1) - epoch))*omega
  
  losses_tot1 = [x/len(train_loader) for x in losses_tot1]
  #losses_tot1 = [x/2 for x in losses_tot1]
  #loss_tot1 = sum(losses_tot1)
  loss_tot1 = losses_tot1[0]
  loss_train.append(loss_tot1.detach())

  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval,loss_val):
  if rank == 0:
    global global_step
    generator.eval()
    losses_tot = []
    with torch.no_grad():
      for batch_idx, (x, x_lengths, y, y_lengths) in enumerate(val_loader):
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

        (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask, l_head_wt, l_qry_wt, l_attn_wt), (attn, logw, logw_) = generator(x, x_lengths, y, y_lengths, gen=False)
        l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
        l_length = commons.duration_loss(logw, logw_, x_lengths)

        loss_gs = [l_mle, l_length]
        loss_g = sum(loss_gs)

        if batch_idx == 0:
          losses_tot = loss_gs
        else:
          losses_tot = [x + y for (x, y) in zip(losses_tot, loss_gs)]

        if batch_idx % hps.train.log_interval == 0:
          logger.info('Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(x), len(val_loader.dataset),
            100. * batch_idx / len(val_loader),
            loss_gs[0].item()))
          logger.info([x.item() for x in loss_gs])
           
    
    losses_tot = [x/len(val_loader) for x in losses_tot]
    #loss_tot = sum(losses_tot)
    loss_tot = losses_tot[0]
    loss_val.append(loss_tot.detach())
    scalar_dict = {"loss/g/total": loss_tot}
    scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_tot)})
    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      scalars=scalar_dict)
    logger.info('====> Epoch: {}'.format(epoch))


if __name__ == "__main__":
  main()
