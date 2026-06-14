__author__ = "Semih Tarik Uenal"

import torch
import pathlib
from .criterion import Criterion
from .wandb_setup import *
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import flow_vis

from evaluate.utils import add_quiver  
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
        
        # Early stopping parameters
        self.early_stopping_patience = cfg_trainer.get('early_stopping_patience', None)
        self.early_stopping_min_delta = cfg_trainer.get('early_stopping_min_delta', 0.0)
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        
        if not self.debug:
            wandb_setup(config)
            self.checkpoint_dir = os.path.join(config['cwd'],cfg_trainer['save_dir'],config['name'])
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.model = model
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config['optimizer']['args']['lr'],
            weight_decay=config['optimizer']['args']['weight_decay'],
            amsgrad=config['optimizer']['args']['amsgrad'],
            eps=config['optimizer']['args']['eps']
        )

        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=config['lr_scheduler']['args']['max_lr'],
            steps_per_epoch=len(self.data_loader),
            epochs=self.num_epochs,
            pct_start=config['lr_scheduler']['args']['pct_start'],
            cycle_momentum=config['lr_scheduler']['args']['cycle_momentum'],
            anneal_strategy=config['lr_scheduler']['args']['anneal_strategy']
)

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
        

    def _save_checkpoint(self, epoch, iter='', is_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param is_best: if True, save as best model
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss
        }
        if iter=='':
            filename = os.path.join(self.checkpoint_dir,f'{epoch}_final.pth')
        else:
            filename = os.path.join(self.checkpoint_dir,f'checkpoint-epoch{epoch}-iter{iter}.pth')

        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving best model: {} ...".format(best_path))

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
        
        # Restore early stopping state if available
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))



def _tensor_to_numpy_01(x):
    # x: torch.Tensor on GPU/CPU, arbitrary dtype
    x = x.detach().float().cpu().numpy()
    # don't normalize here; flow_vis takes raw vectors
    return x

def make_flow_preview_xy_zy(flow_bchwDHW, fixed_b1DHW, step_idx, z_idx=None, x_idx=None):
    """
    flow: (B,3,D,H,W) as [dz,dy,dx]
    """
    B, C, D, H, W = flow_bchwDHW.shape
    assert C == 3
    if z_idx is None: z_idx = D // 2
    if x_idx is None: x_idx = W // 2

    flow = _tensor_to_numpy_01(flow_bchwDHW[0])   # (3,D,H,W)
    img  = _tensor_to_numpy_01(fixed_b1DHW[0])[0] # (D,H,W)

    dz, dy, dx = flow[0], flow[1], flow[2]

    # --- XY @ fixed z (rows=H, cols=W). components: (dx,dy) ---
    u_xy = np.stack([dx[z_idx], dy[z_idx]], axis=-1)   # (H,W,2)
    flow_xy_rgb = flow_vis.flow_to_color(u_xy, convert_to_bgr=False)
    img_xy = img[z_idx]                                # (H,W)

    # --- ZY @ fixed x (rows=D=z, cols=H=y). components: (dz,dy) ---
    u_zy = np.stack([dz[..., x_idx], dy[..., x_idx]], axis=-1)  # (D,H,2)
    flow_zy_rgb = flow_vis.flow_to_color(u_zy, convert_to_bgr=False)
    img_zy = img[..., x_idx]                                   # (D,H)

    return {
        "flow_xy_rgb": flow_xy_rgb,
        "img_xy": img_xy,
        "u_xy": u_xy,                 # <--- for quiver (H,W,2)

        "flow_zy_rgb": flow_zy_rgb,
        "img_zy": img_zy,
        "u_zy": u_zy,                 # <--- for quiver (D,H,2)

        "z_idx": z_idx,
        "x_idx": x_idx,
        "shape": (D,H,W),
        "step": step_idx,
    }


def log_flow_previews_wandb(preview, tag_prefix="train", stride=8, scale=40, alpha=0.6):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- XY ---
    axes[0].set_title(f'XY @ z={preview["z_idx"]}')
    axes[0].imshow(preview["img_xy"], cmap='gray')
    # add quiver: expects u[...,0]=ux, u[...,1]=uy ; it negates uy internally
    add_quiver(axes[0], preview["u_xy"], stride=stride, scale=scale)
    axes[0].axis('off')

    # --- ZY ---
    axes[1].set_title(f'ZY @ x={preview["x_idx"]}')
    axes[1].imshow(preview["img_zy"], cmap='gray', aspect='auto')
    add_quiver(axes[1], preview["u_zy"], stride=stride, scale=scale)
    axes[1].axis('off')

    plt.tight_layout()

    wandb.log({
        f"{tag_prefix}/flow_panels": wandb.Image(fig),
        f"{tag_prefix}/flow_xy": wandb.Image(preview["flow_xy_rgb"]),
        f"{tag_prefix}/flow_zy": wandb.Image(preview["flow_zy_rgb"]),
    })
    plt.close(fig)


class Trainer(BaseTrainer):
    def __init__(self, config, model, data_loader):
        super().__init__(config, model, data_loader)

    def run(self):
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            FREQ = 10
            FREQ_save = 100
            FREQ_IMG = 100   # log images less often than scalars

            train_log_dic = {}
            epoch_loss = 0.0
            steps = 0
            for batch_idx, (data_blob, target) in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                ref_img, mov_img = [x.cuda() for x in target]                   # (B,1,D,H,W)
                
                # Use mixed precision for forward pass
                with torch.cuda.amp.autocast(enabled=self.config['trainer']['use_mixed_precision']):
                    flow_predictions = self.model(ref_img, mov_img)             # list or (B,3,D,H,W)
                    loss_dic = self.criterion(flow_predictions, [ref_img, mov_img])
                    loss = loss_dic['total_loss']
                
                self.backwards(loss)
                epoch_loss += loss.item()

                # accumulate scalar logs
                for key in loss_dic:
                    train_log_dic[key] = train_log_dic.get(key, 0) + loss_dic[key].item()

                # periodic scalar logging
                if steps % FREQ == FREQ - 1:
                    train_log_dic = {k: v / FREQ for k, v in train_log_dic.items()}
                    print(f'Epoch {epoch} [{batch_idx}/{len(self.data_loader)}]', train_log_dic)
                    if not self.debug:
                        wandb.log(train_log_dic)
                    train_log_dic = {}

                # periodic image logging
                if (not self.debug) and (steps % FREQ_IMG == 0):
                    # pick the highest-res flow prediction
                    if isinstance(flow_predictions, (list, tuple)):
                        flow_pred = flow_predictions[-1]
                    else:
                        flow_pred = flow_predictions
                    # ensure shape (B,3,D,H,W)
                    assert flow_pred.dim() == 5 and flow_pred.size(1) == 3

                    # your inputs are already normalized like dataloader; if not, normalize here
                    preview = make_flow_preview_xy_zy(
                        flow_bchwDHW=flow_pred, 
                        fixed_b1DHW=ref_img,     # for background panels
                        step_idx=steps,
                        z_idx=None,              # mid-slice by default
                        x_idx=None
                    )
                    log_flow_previews_wandb(preview, tag_prefix="train", stride=8, scale=40, alpha=0.6)

                steps += 1

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(self.data_loader)
            
            # Check for improvement
            is_best = False
            if avg_epoch_loss < self.best_loss - self.early_stopping_min_delta:
                self.best_loss = avg_epoch_loss
                self.epochs_no_improve = 0
                is_best = True
            else:
                self.epochs_no_improve += 1
            
            # end of epoch logging
            print(f'Epoch-{epoch} completed. LR: {self.optimizer.param_groups[0]["lr"]:.2e}, Avg Loss: {avg_epoch_loss:.4f}, Best: {self.best_loss:.4f}')
            
            if not self.debug:
                wandb.log({'epoch': epoch, 'avg_loss': avg_epoch_loss, 'best_loss': self.best_loss})
                self._save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping check
            if self.early_stopping_patience is not None and self.epochs_no_improve >= self.early_stopping_patience:
                print(f'Early stopping triggered after {epoch} epochs. No improvement for {self.epochs_no_improve} epochs.')
                if not self.debug:
                    wandb.log({'early_stopped': True, 'stopped_epoch': epoch})
                break
