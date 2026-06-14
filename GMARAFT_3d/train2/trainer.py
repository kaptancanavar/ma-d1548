__author__ = "Semih Tarik Uenal"

import torch
import pathlib
from .criterion import Criterion
from .wandb_setup import *
from tqdm import tqdm
import os
from .losses import *
from .warp import warp_torch,warp_3d_torch
from preprocessing import MetaImageIO
import sys
from .image_sim import NCC_gauss, NCC_vxm
import optuna
# sys.path.append(
#     r"C:\Users\z0043wnf\AppData\Local\miniforge3\Lib\site-packages\bspline_interp-0.0.0-py3.12-win-amd64.egg"
# )
import bspline_interp , bspline_interp_withgrad 
torch.autograd.set_detect_anomaly(True)
def pad_image_for_bspline(img: torch.Tensor, padding: int = 2) -> torch.Tensor:
    """
    Pad (Z, Y, X) axes symmetrically for cubic B-spline support.
    """
    return F.pad(img, pad=(padding, padding, padding, padding, padding, padding), mode='reflect')


def warp_3d_batch_bspline(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp a 3D complex-valued image using B-spline interpolation and voxel-space motion fields.

    Args:
        img (torch.Tensor): Complex input image of shape (M, D, H, W), where:
            - M: number of motion states
            - D: depth (Lin)
            - H: height (Par)
            - W: width (Col)
        flow (torch.Tensor): Voxel-space displacement fields of shape (M, D, H, W, 3),
            where the last dimension corresponds to (ΔZ, ΔY, ΔX) displacements.

    Returns:
        torch.Tensor: Warped complex image of shape (M, D, H, W)
    """

    pad = 2
    img = img.squeeze(0)
    flow = flow.permute(0,2,3,4,1).contiguous()
    print(img.shape)
    M, D, H, W = img.shape  # Unpack dimensions: motion_states, Lin, Par, Col

    img_real = pad_image_for_bspline(img, pad)

    warped_real = torch.zeros_like(img)

    bspline_interp.bspline_prefilter(img_real.contiguous())

    z = torch.arange(D, device=img.device)
    y = torch.arange(H, device=img.device)
    x = torch.arange(W, device=img.device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    grid = torch.stack((zz, yy, xx), dim=-1).float().unsqueeze(0).repeat(M, 1, 1, 1, 1)

    coords = grid + flow + pad

    bspline_interp.bspline_interp(img_real.contiguous(), coords.contiguous(), warped_real)

    return warped_real.unsqueeze(0)

# def warp_autograd(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
#     pad = 2
#     img = img.squeeze(0)
#     # 1) Permute & reorder flow to match (z,y,x)
#     flow = flow.permute(0,2,3,4,1).contiguous()  # (M,D,H,W,3) in (dx,dy,dz)                   # now (dz,dy,dx)

#     M, D, H, W = img.shape

#     # 2) Pad & prefilter
#     img_real = pad_image_for_bspline(img, pad)
#     img_real = bspline_interp_withgrad.bspline_prefilter_autograd(
#         img_real.contiguous()
#     )

#     # 3) Build a (z,y,x) grid
#     z = torch.arange(D, device=img.device)
#     y = torch.arange(H, device=img.device)
#     x = torch.arange(W, device=img.device)
#     zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
#     grid = torch.stack((zz, yy, xx), dim=-1)  # (D,H,W,3)
#     grid = grid.unsqueeze(0).repeat(M,1,1,1,1).float()

#     # 4) Combine
#     coords = grid + flow + pad

#     # 5) Interpolate
#     warped_real = bspline_interp_withgrad.bspline_interp_autograd(
#         img_real.contiguous(),
#         coords.contiguous()
#     )

#     return warped_real.unsqueeze(0)

def warp_autograd(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    pad = 2
    img = img.squeeze(0)  # (1,D,H,W) → (M,D,H,W)

    M, D, H, W = img.shape

    # flow: (M, 3, D, H, W) with channels in (x,y,z) order (from coords_grid_3d)
    flow = flow.permute(0, 2, 3, 4, 1).contiguous()  # → (M,D,H,W,3) as (x,y,z)
    
    # CRITICAL: Reorder flow from (x,y,z) to (z,y,x) for B-spline convention
    flow = flow.flip(-1)  # Now (z,y,x)

    # Pad & prefilter
    img_real = pad_image_for_bspline(img, pad)
    img_real = bspline_interp_withgrad.bspline_prefilter_autograd(
        img_real.contiguous()
    )

    # Build identity grid in voxel space in (z,y,x) order for B-spline
    z = torch.arange(D, device=img.device)
    y = torch.arange(H, device=img.device)
    x = torch.arange(W, device=img.device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    grid = torch.stack((zz, yy, xx), dim=-1).unsqueeze(0).float()  # (1,D,H,W,3) as (z,y,x)
    grid = grid.repeat(M, 1, 1, 1, 1)

    # Combine: coords = (z,y,x) + (dz,dy,dx) + pad
    coords = grid + flow + pad

    # Interpolate
    warped_real = bspline_interp_withgrad.bspline_interp_autograd(
        img_real.contiguous(),
        coords.contiguous()
    )

    return warped_real.unsqueeze(0)


class BaseTrainer:
    def __init__(self, config, model, data_loader):
        self.config = config
        self.data_loader = data_loader
        self.debug = config['debug']

        cfg_trainer = config['trainer']
        self.start_epoch = 0
        self.num_epochs = cfg_trainer['epochs']
        self.group = cfg_trainer['group']
        self.clip = cfg_trainer['clip']
        self.gamma = cfg_trainer['gamma']
        self.loss_function = PhotometricLoss().to("cuda")
        
        if not self.debug:
            wandb_setup(config)
            self.checkpoint_dir = os.path.join(config['cwd'],cfg_trainer['save_dir'],config['name'])
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.model = model
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = torch.optim.AdamW(trainable_params,
                                           lr = config['optimizer']['args']['lr'],
                                           weight_decay = config['optimizer']['args']['weight_decay'],
                                           amsgrad=config['optimizer']['args']['amsgrad'],
                                           eps=config['optimizer']['args']['eps'])



        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer,
                                                                max_lr=0.00025,
                                                                steps_per_epoch=len(self.data_loader),
                                                                epochs=self.num_epochs,
                                                                pct_start=0.05,
                                                                cycle_momentum=False,
                                                                anneal_strategy='linear')

        self.criterion = Criterion(config['loss_functions']['args'])

        if cfg_trainer['resume']:
            self._resume_checkpoint(cfg_trainer['resume'])

        self.loss_scaler = torch.cuda.amp.GradScaler(enabled=cfg_trainer['use_mixed_precision'])

    def backwards(self, loss):
        self.loss_scaler.scale(loss).backward()
        self.loss_scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.loss_scaler.step(self.optimizer)
        self.lr_scheduler.step()
        self.loss_scaler.update()
        

    def _save_checkpoint(self, epoch, iter=''):
        """
        Saving checkpoints
        :param epoch: current epoch number
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        if iter=='':
            filename = os.path.join(self.checkpoint_dir,f'{epoch}_final.pth')
        else:
            filename = os.path.join(self.checkpoint_dir,f'checkpoint-epoch{epoch}-iter{iter}.pth')

        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


# class Trainer(BaseTrainer):
#     def __init__(self, config , model, data_loader):
#         super().__init__(config , model, data_loader)
        
#     def run(self):
#         for epoch in range(self.start_epoch, self.num_epochs + 1):
#             FREQ = 1
#             FREQ_save = 500
#             train_log_dic = {}
#             steps = 0
#             for batch_idx, (data_blob, target) in tqdm(enumerate(self.data_loader), total=len(self.data_loader), desc=f"Epoch {epoch}"):
#                 self.optimizer.zero_grad()
#                 ref_img, mov_img, context_img = [x.cuda() for x in data_blob]
#                 ref, mov, context = [x.cuda() for x in target]
#                 flow_predictions_list, context_pred = self.model(ref_img, mov_img, context_img)

#                 # loss_dic = self.criterion(flow_predictions, target)
#                 # loss = loss_dic['total_loss']
#                 self.optimizer.zero_grad()
#                 flow_predictions = [flow_predictions_list[-1]]
#                 total_iters = len(flow_predictions)
#                 device = mov.device  
#                 total_loss = torch.tensor(0.0, device=device) 
#                 for flow in flow_predictions:
#                     warped = warp_3d_batch_bspline(mov, flow)
#                     total_loss += F.l1_loss(warped, ref) / total_iters

#                 total_loss.backward()
#                 self.optimizer.step()
#                 self.lr_scheduler.step()
                
                
#                 # for key in loss_dic:
#                 #     if key not in train_log_dic.keys():
#                 #         train_log_dic[key] = 0
#                 #     train_log_dic[key] += loss_dic[key].item()
#                 train_log_dic["L1"] = total_loss.item()
#                 if steps % FREQ == FREQ - 1:
#                     train_log_dic = {k: v / FREQ for k, v in train_log_dic.items()}
#                     print(f'{batch_idx}/{len(self.data_loader)}', train_log_dic)
#                     # print('loss', f'{batch_idx}/{len(self.data_loader)}', loss.item())
#                     #
#                     if not self.debug:
#                         print(print("Logging to wandb:", train_log_dic))
#                         wandb.log(train_log_dic)
#                     train_log_dic = {}
#                 if steps % FREQ_save == FREQ_save - 1:
#                     self._save_checkpoint(epoch, steps)

#                 steps += 1

#             train_log_dic = {k: v / len(self.data_loader) for k, v in train_log_dic.items()}
#             # train_log_dic['lr'] = scheduler.get_last_lr()

#             print('Epoch-{0} lr: {1}'.format(epoch, self.optimizer.param_groups[0]['lr']))
#             print('train logs:\n', train_log_dic)

#             if not self.debug:
#                 self._save_checkpoint(epoch)

class Trainer(BaseTrainer):
    def __init__(self, config , model, data_loader):
        super().__init__(config , model, data_loader)
        self.debug_overfit = True
        self.loss = NCC_vxm().to("cuda")
        # Regularization weights - reduced for faster convergence
        self.alpha_smooth = 0.001      # Very light smoothness
        self.alpha_magnitude = 0.0001 # Very light magnitude penalty
            
    def compute_flow_regularization(self, flow):
        """
        Compute smoothness and magnitude regularization for flow field.
        flow: (B, 3, D, H, W) tensor
        """
        # Smoothness: penalize spatial gradients
        flow_grad_x = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
        flow_grad_y = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
        flow_grad_z = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
        smoothness_loss = (flow_grad_x.abs().mean() + 
                          flow_grad_y.abs().mean() + 
                          flow_grad_z.abs().mean())
        
        # Magnitude: penalize large displacements
        magnitude_loss = flow.abs().mean()
        
        return smoothness_loss, magnitude_loss
    
    def run(self):
        
        if self.debug_overfit:
            train_log_dic = {}
            # Grab exactly one batch and loop on it
            data_blob, target = next(iter(self.data_loader))
            # ref_img, mov_img, context_img = [x.cuda() for x in data_blob]

            # ref, mov, context = [x.cuda() for x in target]
            ref_img, mov_img = [x.cuda() for x in data_blob]
            # ref, mov = [x.cuda() for x in target]
            # Zero everything
            self.optimizer.zero_grad()
            freq = 1

            ref_np    = ref_img.detach().cpu().numpy()[0,0]   
            mov_np    =  mov_img.detach().cpu().numpy()[0,0]      
            out_dir = os.path.join(os.getcwd(), "debug_overfit")
            os.makedirs(out_dir, exist_ok=True)

            print("Writing out debug volumes to", out_dir)
            MetaImageIO.write(os.path.join(out_dir, "fixed.mhd"), ref_np)
            MetaImageIO.write(os.path.join(out_dir, "moving.mhd"), mov_np)

            print("=== OVERFIT MODE: single‐example loop ===")
            for step in range(500):                  # run 500 updates
                flow_predictions= self.model(ref_img, mov_img)

                total_loss = 0.0
                flow_pred = [flow_predictions[-1]]

                for itr, flow in enumerate(flow_pred):
                    if step % freq == 0:
                        print(f"[step {step}] flow0 min/max:",
                              float(flow.min()), float(flow.max()))
                    
                    warped = warp_autograd(mov_img, flow)
                    
                    # Photometric loss
                    photometric_loss = self.loss(ref_img, warped)
                    
                    # Regularization
                    smoothness_loss, magnitude_loss = self.compute_flow_regularization(flow)
                    
                    # Combined loss
                    l = (photometric_loss + 
                         self.alpha_smooth * smoothness_loss + 
                         self.alpha_magnitude * magnitude_loss)
                    
                    if step % freq == 0:
                        print(f"   iter{itr} photometric: {float(photometric_loss):.4f}, "
                              f"smoothness: {float(smoothness_loss):.4f}, "
                              f"magnitude: {float(magnitude_loss):.4f}")
                    
                    total_loss += l / len(flow_pred)

                # Backprop + step
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if step % freq == 0:
                    print(f"[step {step}] total_loss = {total_loss.item():.6f}")
                    train_log_dic["total_loss"] = total_loss.item()
                    train_log_dic["photometric"] = float(photometric_loss)
                    train_log_dic["smoothness"] = float(smoothness_loss)
                    train_log_dic["magnitude"] = float(magnitude_loss)
                    wandb.log(train_log_dic)



                warped_np = warped.detach().cpu().numpy()[0,0]  # (D,H,W)

                MetaImageIO.write(os.path.join(out_dir, "warped.mhd"), warped_np)

            print("Done writing. Exiting overfit debug.")
            return
        train_log_dic = {}
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            steps = 0
            for batch_idx, (data_blob, target) in tqdm(enumerate(self.data_loader),
                                                      total=len(self.data_loader),
                                                      desc=f"Epoch {epoch}"):
                # Unpack
                data_blob, target = next(iter(self.data_loader))
                # ref_img, mov_img, context_img = [x.cuda() for x in data_blob]

                # ref, mov, context = [x.cuda() for x in target]
                ref_img, mov_img = [x.cuda() for x in data_blob]

                # Zero grads
                self.optimizer.zero_grad()

                # Forward
                flow_pred = self.model(ref_img, mov_img)
                flow_predictions = [flow_pred[-1]]
                # DEBUG #1: print shapes at every stage
                if steps == 0:
                    print(" ref_img:", tuple(ref_img.shape))
                    print(" mov_img:", tuple(mov_img.shape))
                    print(" fmap1:", tuple(self.model.fnet([ref_img, mov_img])[0].shape))
                    print(" coords0:", tuple(self.model.initialize_flow(ref_img)[0].shape))
                    print(" #iterations:", len(flow_predictions))
                
                # Photometric loss over each refinement
                total_loss = 0.0
                for itr, flow in enumerate(flow_predictions):
                    # DEBUG #2: track min/max of the raw flows
                    if itr == 0 and steps % 1 == 0:
                        print(f"  iter0 flow min/max:", flow.min().item(), flow.max().item())

                    warped = warp_autograd(mov_img, flow)
                    # l = F.mse_loss(ref, warped)
                    l = self.loss(ref_img,warped)
                    # DEBUG #3: track photometric error occasionally
                    if steps ==0 and steps % 1 == 0:
                        err_map = (warped - ref_img).abs()
                        print(f"   photometric L1  iter{itr}:", err_map.mean().item())

                    total_loss += l / len(flow_predictions)

                # Backward + step
                total_loss.backward()

                # DEBUG #4: gradient norms
                if steps % 50 == 0:
                    total_norm = 0.0
                    for name, p in self.model.named_parameters():
                        if p.grad is not None:
                            gnorm = p.grad.data.norm(2).item()
                            total_norm += gnorm**2
                    print("  ∥grads∥:", total_norm**0.5)

                self.optimizer.step()
                self.lr_scheduler.step()

                # Logging
                if steps % 1 == 0:
                    train_log_dic["L1"] = total_loss.item()
                    wandb.log(train_log_dic)
                steps += 1

            # epoch‐end checkpoint
            print(f"Epoch {epoch} finished, saving checkpoint.")
            self._save_checkpoint(epoch)

# class Trainer(BaseTrainer):
#     # Hard-coded hyperparameters
#     DEFAULT_LR = 1e-3
#     DEFAULT_WD = 1e-4
#     SCHED_STEP_SIZE = 10
#     SCHED_GAMMA = 0.5
#     # Optuna tuning parameters
#     OPTUNA_ENABLED = True
#     OPTUNA_N_TRIALS = 50
#     OPTUNA_TIMEOUT_SECS = 3600  # 1 hour
#     OPTUNA_STEPS_PER_TRIAL = 100

#     def __init__(self, model, data_loader, num_epochs=100, start_epoch=1):
#         super().__init__({'optimizer': {}, 'scheduler': {}}, model, data_loader)
#         self.loss_fn = NCC_gauss()
#         self.num_epochs = num_epochs
#         self.start_epoch = start_epoch
#         # initialize optimizer and scheduler with defaults
#         self._init_optim(lr=self.DEFAULT_LR, wd=self.DEFAULT_WD)

#     def _init_optim(self, lr=None, wd=None):
#         lr = lr if lr is not None else self.DEFAULT_LR
#         wd = wd if wd is not None else self.DEFAULT_WD
#         self.optimizer = torch.optim.Adam(
#             self.model.parameters(), lr=lr, weight_decay=wd
#         )
#         self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
#             self.optimizer,
#             step_size=self.SCHED_STEP_SIZE,
#             gamma=self.SCHED_GAMMA
#         )

#     def _training_step(self, ref_img, mov_img, ref, mov):
#         flow_pred, _ = self.model(ref_img, mov_img, None)
#         flow = flow_pred[-1]
#         warped = warp_autograd(mov, flow)
#         return F.mse_loss(ref, warped)

#     def _objective(self, trial):
#         # Suggest hyperparameters
#         lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
#         wd = trial.suggest_loguniform("wd", 1e-6, 1e-3)

#         # Reset weights and optimizer with sampled values
#         self.model.apply(self.model.init_weights)
#         self._init_optim(lr=lr, wd=wd)

#         data_iter = iter(self.data_loader)
#         total_loss = 0.0
#         for step in range(self.OPTUNA_STEPS_PER_TRIAL):
#             try:
#                 data_blob, target = next(data_iter)
#             except StopIteration:
#                 data_iter = iter(self.data_loader)
#                 data_blob, target = next(data_iter)

#             ref_img, mov_img = [x.cuda() for x in data_blob[:2]]
#             ref, mov = [x.cuda() for x in target[:2]]

#             self.optimizer.zero_grad()
#             loss = self._training_step(ref_img, mov_img, ref, mov)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#             self.optimizer.step()
#             self.lr_scheduler.step()

#             total_loss += loss.item()
#             trial.report(loss.item(), step)
#             if trial.should_prune():
#                 raise optuna.TrialPruned()

#         return total_loss / self.OPTUNA_STEPS_PER_TRIAL

#     def run_optuna(self):
#         if not self.OPTUNA_ENABLED:
#             return None
#         study = optuna.create_study(
#             direction="minimize",
#             pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
#         )
#         study.optimize(
#             self._objective,
#             n_trials=self.OPTUNA_N_TRIALS,
#             timeout=self.OPTUNA_TIMEOUT_SECS
#         )
#         print("Optuna completed. Best params:")
#         print(study.best_params)
#         # Re-init with best
#         best = study.best_params
#         self._init_optim(lr=best.get('lr', self.DEFAULT_LR), wd=best.get('wd', self.DEFAULT_WD))
#         return study.best_params

#     def run(self):
#         # Hyperparameter search first
#         if self.OPTUNA_ENABLED:
#             best_params = self.run_optuna()
#             if best_params:
#                 wandb.config.update(best_params)

#         # Full training loop
#         for epoch in range(self.start_epoch, self.num_epochs + 1):
#             for _, (data_blob, target) in enumerate(self.data_loader):
#                 ref_img, mov_img = [x.cuda() for x in data_blob[:2]]
#                 ref, mov = [x.cuda() for x in target[:2]]

#                 self.optimizer.zero_grad()
#                 loss = self._training_step(ref_img, mov_img, ref, mov)
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#                 self.optimizer.step()
#                 self.lr_scheduler.step()

#                 wandb.log({'loss': loss.item()})

#             print(f"Epoch {epoch} done.")
#             self._save_checkpoint(epoch)
