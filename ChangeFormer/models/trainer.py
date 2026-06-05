import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import json

import utils
from models.networks import *

import torch
import torch.optim as optim
import numpy as np
from misc.metric_tool import ConfuseMatrixMeter
from models.checkpoint_utils import (
    extract_model_state_dict,
    load_model_state_dict,
    torch_load_checkpoint,
)
from models.losses import cross_entropy
import models.losses as losses
from models.losses import get_alpha, softmax_helper, FocalLoss, mIoULoss, mmIoULoss

from misc.logger_tool import Logger, Timer

from utils import de_norm

from tqdm import tqdm

import torch.nn.functional as F 

class CDTrainer():

    def __init__(self, args, dataloaders):
        self.args = args
        self.dataloaders = dataloaders

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)


        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr

        # define optimizers
        if args.optimizer == "sgd":
            self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                     momentum=0.9,
                                     weight_decay=5e-4)
        elif args.optimizer == "adam":
            self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr,
                                     weight_decay=0)
        elif args.optimizer == "adamw":
            self.optimizer_G = optim.AdamW(self.net_G.parameters(), lr=self.lr,
                                    betas=(0.9, 0.999), weight_decay=0.01)

        # self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr)

        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        #ben ekledim
        self.patience = args.patience
        self.epochs_no_improve = 0
        self.early_stop = False
        #
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        self.shuffle_AB = args.shuffle_AB

        # define the loss functions
        self.multi_scale_train = args.multi_scale_train
        self.multi_scale_infer = args.multi_scale_infer
        self.weights = tuple(args.multi_pred_weights)
        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
        elif args.loss == 'bce':
            self._pxl_loss = losses.binary_ce
        elif args.loss == 'fl':
            print('\n Calculating alpha in Focal-Loss (FL) ...')
            alpha           = get_alpha(dataloaders['train']) # calculare class occurences
            print(f"alpha-0 (no-change)={alpha[0]}, alpha-1 (change)={alpha[1]}")
            self._pxl_loss  = FocalLoss(apply_nonlin = softmax_helper, alpha = alpha, gamma = 2, smooth = 1e-5)
        elif args.loss == "miou":
            print('\n Calculating Class occurances in training set...')
            alpha   = np.asarray(get_alpha(dataloaders['train'])) # calculare class occurences
            alpha   = alpha/np.sum(alpha)
            # weights = torch.tensor([1.0, 1.0]).cuda()
            weights = 1-torch.from_numpy(alpha).cuda()
            print(f"Weights = {weights}")
            self._pxl_loss = mIoULoss(weight=weights, size_average=True, n_classes=args.n_class).cuda()
        elif args.loss == "mmiou":
            self._pxl_loss = mmIoULoss(n_classes=args.n_class).cuda()
        ##            
        elif args.loss == "weighted_ce":                     #ben ekledim,
            weight = torch.tensor([1.0, 20.5], device =self.device)
            _ce = nn.CrossEntropyLoss(weight=weight, ignore_index=255)

            def weighted_ce(pred,target):
                if target.dim() == 4:
                    target = target.squeeze(1)
                if pred.shape[-1] != target.shape[-1]:
                    pred = F.interpolate(pred, size=target.shape[1:],mode='bilinear', align_corners=True)
                return _ce(pred, target.long())
            
            self._pxl_loss = weighted_ce
        ##
        elif args.loss == "ohem":                            #ben ekledim
            from models.losses import OhemCrossEntropy
            ohem_thresh = getattr(args, 'ohem_thresh', 0.7)
            ohem_ratio  = getattr(args, 'ohem_ratio', 0.5)
            print('\n Using OHEM Cross-Entropy: thresh=%.3f, keep_ratio=%.3f'
                  % (ohem_thresh, ohem_ratio))
            self._pxl_loss = OhemCrossEntropy(
                thresh=ohem_thresh,
                keep_ratio=ohem_ratio,
                ignore_index=255,
            ).to(self.device)
        ##   
        else:
            raise NotImplemented(args.loss)

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        # ── per-epoch history for YOLO-style plots ──
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {
                'train_loss': [], 'train_mf1': [], 'train_miou': [],
                'val_mf1': [], 'val_miou': [],
                'val_precision_0': [], 'val_precision_1': [],
                'val_recall_0': [], 'val_recall_1': [],
                'val_f1_0': [], 'val_f1_1': [],
                'lr': [], 'epoch_time_min': [],
            }
        self._epoch_loss_sum = 0.0
        self._epoch_loss_count = 0

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)
    


    def _load_training_checkpoint(self, ckpt_path):
        finetune = getattr(self.args, 'finetune', False)
        mode_label = 'FINE-TUNING' if finetune else 'RESUMING'
        self.logger.write('=== %s from checkpoint: %s ===\n' % (mode_label, ckpt_path))

        # load checkpoint file
        checkpoint = torch_load_checkpoint(ckpt_path, map_location=self.device, logger=self.logger)

        # load model weights (always)
        model_state_dict, model_state_key = extract_model_state_dict(checkpoint)
        load_model_state_dict(self.net_G, model_state_dict, strict=True)
        self.net_G.to(self.device)
        self.logger.write('Loaded model weights from key: %s\n' % model_state_key)

        if finetune:
            # fine-tune mode: only weights, everything else fresh
            self.logger.write('[Fine-tune] Starting fresh optimizer/scheduler/epoch.\n')
            self.epoch_to_start = 0
            self.best_val_acc = 0.0
            self.best_epoch_id = 0
        else:
            # resume mode: load optimizer, scheduler, epoch, best score
            if 'optimizer_G_state_dict' in checkpoint:
                self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
                self.logger.write('[Resume] Loaded optimizer state.\n')

            if 'exp_lr_scheduler_G_state_dict' in checkpoint:
                self.exp_lr_scheduler_G.load_state_dict(checkpoint['exp_lr_scheduler_G_state_dict'])
                self.logger.write('[Resume] Loaded scheduler state.\n')

            if 'epoch_id' in checkpoint:
                self.epoch_to_start = int(checkpoint['epoch_id']) + 1
                self.logger.write('[Resume] Continuing from epoch %d.\n' % self.epoch_to_start)

            if 'best_val_acc' in checkpoint:
                self.best_val_acc = float(checkpoint['best_val_acc'])

            if 'best_epoch_id' in checkpoint:
                self.best_epoch_id = int(checkpoint['best_epoch_id'])

            self.logger.write('[Resume] best_val_acc: %.4f (epoch %d)\n'
                              % (self.best_val_acc, self.best_epoch_id))

        # recalculate total steps and print summary
        self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch
        self.logger.write('Mode: %s, start epoch: %d, total_steps: %d\n'
                          % (mode_label, self.epoch_to_start, self.total_steps))

    def _optimizer_state_is_compatible(self, optimizer_state, checkpoint_optimizer=None):
        if not isinstance(optimizer_state, dict):
            return False, 'optimizer state is not a dict'

        param_groups = optimizer_state.get('param_groups')
        if not param_groups:
            return False, 'optimizer state has no param_groups'

        if checkpoint_optimizer is not None and checkpoint_optimizer != self.args.optimizer:
            return False, 'checkpoint optimizer is %s but args.optimizer is %s' % (
                checkpoint_optimizer, self.args.optimizer)

        required_keys = set(self.optimizer_G.defaults.keys()) | {'lr'}
        saved_keys = set(param_groups[0].keys())
        missing_keys = sorted(required_keys - saved_keys)
        if missing_keys:
            return False, 'missing optimizer param-group keys: %s; use the same --optimizer as the checkpoint' % (
                ', '.join(missing_keys))

        return True, ''

    def _load_pretrain(self):
        print("Initializing backbone weights from: " + self.args.pretrain)
        checkpoint = torch_load_checkpoint(self.args.pretrain, map_location=self.device, logger=self.logger)
        model_state_dict, model_state_key = extract_model_state_dict(checkpoint)
        load_info, prefix_action = load_model_state_dict(
            self.net_G, model_state_dict, strict=False)
        self.net_G.to(self.device)
        self.net_G.eval()
        self.logger.write('Loaded pretrained weights from key: %s (%s)\n'
                          % (model_state_key, prefix_action))
        if hasattr(load_info, 'missing_keys') and hasattr(load_info, 'unexpected_keys'):
            self.logger.write('Pretrain load missing keys: %d, unexpected keys: %d\n'
                              % (len(load_info.missing_keys), len(load_info.unexpected_keys)))

    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):
        print("\n")
        resume_path = getattr(self.args, 'resume_path', None)
        default_ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)

        if resume_path is not None:
            if not os.path.exists(resume_path):
                raise FileNotFoundError('no such resume checkpoint %s' % resume_path)
            self._load_training_checkpoint(resume_path)
        elif os.path.exists(default_ckpt_path):
            self.logger.write('Loading checkpoint from checkpoint_dir: %s\n' % default_ckpt_path)
            self._load_training_checkpoint(default_ckpt_path)
        elif getattr(self.args, 'resume', False):
            raise FileNotFoundError('resume requested, but no such checkpoint %s' % default_ckpt_path)
        elif self.args.pretrain is not None:
            self._load_pretrain()
        else:
            print('training from scratch...')
        print("\n")

    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _visualize_pred(self):
        pred = torch.argmax(self.G_final_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_name': self.args.optimizer,
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
            'lr_policy': self.args.lr_policy,
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_final_pred.detach()

        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' %\
                      (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                     imps*self.batch_size, est,
                     self.G_loss.item(), running_acc)
            self.logger.write(message)


        if np.mod(self.batch_id, 500) == 1:
            vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

            vis_pred = utils.make_numpy_grid(self._visualize_pred())

            vis_gt = utils.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'istrain_'+str(self.is_training)+'_'+
                              str(self.epoch_id)+'_'+str(self.batch_id)+'.jpg')
            plt.imsave(file_name, vis)

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')
        self.logger.write('\n')

    def _update_checkpoints(self):
        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
            % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self.epochs_no_improve = 0
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')
        else:
            self.epochs_no_improve += 1
            self.logger.write('No improvement for %d epoch(s). Patience: %d/%d\n' %
                            (self.epochs_no_improve, self.epochs_no_improve, self.patience))
            self.logger.write('\n')
            if self.epochs_no_improve >= self.patience:
                self.logger.write('Early stopping triggered at epoch %d.\n' % self.epoch_id)
                self.early_stop = True
    
    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _save_confusion_matrix(self):
        """Save the confusion matrix from the last validation epoch as a heatmap."""
        cm = self.running_metric.sum  # accumulated confusion matrix
        if cm is None:
            return

        # normalize each row so values are fractions of the actual class
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)

        # set up the figure
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # axis labels and ticks
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['No Change', 'Change'])
        ax.set_yticklabels(['No Change', 'Change'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Normalized Confusion Matrix')

        # top-left cell: actual No Change, predicted No Change (TN)
        count = int(cm[0, 0])
        pct = cm_norm[0, 0] * 100
        text_color = 'white' if cm_norm[0, 0] > 0.5 else 'black'
        ax.text(0, 0, f'{count}\n({pct:.1f}%)',
                ha='center', va='center', color=text_color, fontsize=10)

        # top-right cell: actual No Change, predicted Change (FP)
        count = int(cm[0, 1])
        pct = cm_norm[0, 1] * 100
        text_color = 'white' if cm_norm[0, 1] > 0.5 else 'black'
        ax.text(1, 0, f'{count}\n({pct:.1f}%)',
                ha='center', va='center', color=text_color, fontsize=10)

        # bottom-left cell: actual Change, predicted No Change (FN)
        count = int(cm[1, 0])
        pct = cm_norm[1, 0] * 100
        text_color = 'white' if cm_norm[1, 0] > 0.5 else 'black'
        ax.text(0, 1, f'{count}\n({pct:.1f}%)',
                ha='center', va='center', color=text_color, fontsize=10)

        # bottom-right cell: actual Change, predicted Change (TP)
        count = int(cm[1, 1])
        pct = cm_norm[1, 1] * 100
        text_color = 'white' if cm_norm[1, 1] > 0.5 else 'black'
        ax.text(1, 1, f'{count}\n({pct:.1f}%)',
                ha='center', va='center', color=text_color, fontsize=10)

        # save and close
        fig.tight_layout()
        fig.savefig(os.path.join(self.checkpoint_dir, 'confusion_matrix.png'), dpi=150)
        plt.close(fig)

    def _plot_training_results(self):
        h = self.history
        n = len(h['train_loss'])
        epochs = list(range(1, n + 1))

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('Training Results — %s' % self.args.project_name,
                     fontsize=16, fontweight='bold', y=0.98)

        # ── Train Loss (top-left) ──
        ax = axes[0][0]
        ax.plot(epochs, h['train_loss'], label='Train Loss', color='#e74c3c', linewidth=1.5)
        ax.set_title('Train Loss', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── mF1 Score (top-middle) ──
        ax = axes[0][1]
        ax.plot(epochs, h['train_mf1'], label='Train', color='#3498db', linewidth=1.5)
        ax.plot(epochs, h['val_mf1'], label='Val', color='#e74c3c', linewidth=1.5)
        # mark the best val point
        best_idx = int(np.argmax(h['val_mf1']))
        best_val = h['val_mf1'][best_idx]
        ax.scatter(epochs[best_idx], best_val, color='#e74c3c',
                   s=60, zorder=5, edgecolors='black', linewidths=0.8)
        ax.annotate(f'{best_val:.4f}',(epochs[best_idx], best_val),textcoords="offset points", xytext=(5, 8),fontsize=8, color='#e74c3c', fontweight='bold')
        ax.set_title('mF1 Score', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mF1')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── mIoU (top-right) ──
        ax = axes[0][2]
        ax.plot(epochs, h['train_miou'], label='Train', color='#3498db', linewidth=1.5)
        ax.plot(epochs, h['val_miou'], label='Val', color='#e74c3c', linewidth=1.5)
        # mark the best val point
        best_idx = int(np.argmax(h['val_miou']))
        best_val = h['val_miou'][best_idx]
        ax.scatter(epochs[best_idx], best_val, color='#e74c3c',
                   s=60, zorder=5, edgecolors='black', linewidths=0.8)
        ax.annotate(f'{best_val:.4f}',
                    (epochs[best_idx], best_val),
                    textcoords="offset points", xytext=(5, 8),
                    fontsize=8, color='#e74c3c', fontweight='bold')
        ax.set_title('mIoU', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mIoU')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Val Precision for Change Class (middle-left) ──
        ax = axes[1][0]
        ax.plot(epochs, h['val_precision_1'], label='Change Class', color='#2ecc71', linewidth=1.5)
        ax.set_title('Val Precision (Change)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Precision')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Val Recall for Change Class (middle-middle) ──
        ax = axes[1][1]
        ax.plot(epochs, h['val_recall_1'], label='Change Class', color='#9b59b6', linewidth=1.5)
        ax.set_title('Val Recall (Change)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Val F1 for Change Class (middle-right) ──
        ax = axes[1][2]
        ax.plot(epochs, h['val_f1_1'], label='Change Class', color='#e67e22', linewidth=1.5)
        ax.set_title('Val F1 (Change)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Learning Rate (bottom-left) ──
        ax = axes[2][0]
        ax.plot(epochs, h['lr'], label='LR', color='#1abc9c', linewidth=1.5)
        ax.set_title('Learning Rate', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('LR')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Epoch Time (bottom-middle) ──
        ax = axes[2][1]
        ax.plot(epochs, h['epoch_time_min'], label='Time', color='#34495e', linewidth=1.5)
        ax.set_title('Epoch Time', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Minutes')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── bottom-right: summary text box ──
        ax_summary = axes[2][2]
        ax_summary.axis('off')
        best_val_mf1 = max(h['val_mf1']) if h['val_mf1'] else 0
        best_val_miou = max(h['val_miou']) if h['val_miou'] else 0
        best_epoch_mf1 = int(np.argmax(h['val_mf1'])) + 1 if h['val_mf1'] else 0
        best_epoch_miou = int(np.argmax(h['val_miou'])) + 1 if h['val_miou'] else 0
        final_train_loss = h['train_loss'][-1] if h['train_loss'] else 0
        total_time = sum(h['epoch_time_min']) if h['epoch_time_min'] else 0

        summary = (
            f"{'═' * 32}\n"
            f"  TRAINING SUMMARY\n"
            f"{'═' * 32}\n"
            f"  Epochs:          {n}\n"
            f"  Best val mF1:    {best_val_mf1:.4f} (ep {best_epoch_mf1})\n"
            f"  Best val mIoU:   {best_val_miou:.4f} (ep {best_epoch_miou})\n"
            f"  Final train loss:{final_train_loss:.5f}\n"
            f"  Total time:      {total_time:.1f} min\n"
            f"{'─' * 32}\n"
            f"  Model:    {self.args.net_G}\n"
            f"  Loss:     {self.args.loss}\n"
            f"  Optimizer:{self.args.optimizer}\n"
            f"  LR:       {self.args.lr}\n"
            f"  Batch:    {self.args.batch_size}\n"
            f"{'═' * 32}"
        )
        ax_summary.text(0.05, 0.95, summary, transform=ax_summary.transAxes,
                        fontsize=10, verticalalignment='top',
                        fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa',
                                  edgecolor='#dee2e6', alpha=0.9))

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        results_path = os.path.join(self.checkpoint_dir, 'results.png')
        fig.savefig(results_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        self.logger.write('Results plot saved: %s\n' % results_path)

    def _clear_cache(self):
        self.running_metric.clear()


    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)

        if self.multi_scale_infer == "True":
            self.G_final_pred = torch.zeros(self.G_pred[-1].size()).to(self.device)
            for pred in self.G_pred:
                if pred.size(2) != self.G_pred[-1].size(2):
                    self.G_final_pred = self.G_final_pred + F.interpolate(pred, size=self.G_pred[-1].size(2), mode="nearest")
                else:
                    self.G_final_pred = self.G_final_pred + pred
            self.G_final_pred = self.G_final_pred/len(self.G_pred)
        else:
            self.G_final_pred = self.G_pred[-1]

            
    def _backward_G(self):
        gt = self.batch['L'].to(self.device).float()
        if self.multi_scale_train == "True":
            i         = 0
            temp_loss = 0.0
            for pred in self.G_pred:
                if pred.size(2) != gt.size(2):
                    temp_loss = temp_loss + self.weights[i]*self._pxl_loss(pred, F.interpolate(gt, size=pred.size(2), mode="nearest"))
                else:
                    temp_loss = temp_loss + self.weights[i]*self._pxl_loss(pred, gt)
                i+=1
            self.G_loss = temp_loss
        else:
            self.G_loss = self._pxl_loss(self.G_pred[-1], gt)

        self.G_loss.backward()


    def train_models(self):

        self._load_checkpoint()

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            epoch_start_time = time.time()

            ################## train #################
            ##########################################
            self._clear_cache()
            self._epoch_loss_sum = 0.0
            self._epoch_loss_count = 0
            self.is_training = True
            self.net_G.train()  # Set model to training mode
            # Iterate over data.
            total = len(self.dataloaders['train'])
            self.logger.write('lr: %0.7f\n \n' % self.optimizer_G.param_groups[0]['lr'])
            for self.batch_id, batch in tqdm(enumerate(self.dataloaders['train'], 0), total=total):
                self._forward_pass(batch)
                # update G
                self.optimizer_G.zero_grad()
                self._backward_G()
                self.optimizer_G.step()
                self._collect_running_batch_states()
                self._timer_update()
                # accumulate loss
                self._epoch_loss_sum += self.G_loss.item()
                self._epoch_loss_count += 1

            self._collect_epoch_states()
            # ── record train metrics ──
            train_scores = self.running_metric.get_scores()
            avg_train_loss = self._epoch_loss_sum / max(self._epoch_loss_count, 1)
            self.history['train_loss'].append(round(avg_train_loss, 6))
            self.history['train_mf1'].append(round(train_scores['mf1'], 6))
            self.history['train_miou'].append(round(train_scores['miou'], 6))
            self.history['lr'].append(round(self.optimizer_G.param_groups[0]['lr'], 8))

            self._update_training_acc_curve()
            self._update_lr_schedulers()



            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass(batch)
                self._collect_running_batch_states()
            self._collect_epoch_states()

            # ── record val metrics ──
            val_scores = self.running_metric.get_scores()
            self.history['val_mf1'].append(round(val_scores['mf1'], 6))
            self.history['val_miou'].append(round(val_scores['miou'], 6))
            self.history['val_precision_0'].append(round(val_scores.get('precision_0', 0), 6))
            self.history['val_precision_1'].append(round(val_scores.get('precision_1', 0), 6))
            self.history['val_recall_0'].append(round(val_scores.get('recall_0', 0), 6))
            self.history['val_recall_1'].append(round(val_scores.get('recall_1', 0), 6))
            self.history['val_f1_0'].append(round(val_scores.get('F1_0', 0), 6))
            self.history['val_f1_1'].append(round(val_scores.get('F1_1', 0), 6))

            epoch_time = (time.time() - epoch_start_time) / 60.0
            self.history['epoch_time_min'].append(round(epoch_time, 2))

            # ── save history to disk every epoch ──
            with open(os.path.join(self.checkpoint_dir, 'training_history.json'), 'w') as f:
                json.dump(self.history, f, indent=2)

            ########### Update_Checkpoints ###########
            ##########################################  
            self._update_val_acc_curve()
            self._update_checkpoints()

            if self.early_stop:
                self.logger.write('Training stopped early. Best acc=%.4f at epoch %d.\n' %
                          (self.best_val_acc, self.best_epoch_id))
                break

        # ── Post-training: generate YOLO-style results plot ──
        self.logger.write('\n=== Generating training results plots ===\n')
        try:
            self._save_confusion_matrix()
            self._plot_training_results()
            self.logger.write('Results saved to: %s\n' % self.checkpoint_dir)
        except Exception as e:
            self.logger.write('Warning: Could not generate plots: %s\n' % str(e))
