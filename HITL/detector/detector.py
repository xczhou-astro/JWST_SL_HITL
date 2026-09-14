import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import stack_module_state, functional_call, vmap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
import joblib
import copy
import time
import shutil
from astropy.coordinates import SkyCoord
import astropy.units as u

from models import LatentClassifier, CudaDataLoader, BinaryFocalLoss
from configurations import config

class SLDetector:
    
    def __init__(self):
        
        # TF32 for faster training (enables TensorFloat-32 on Ampere+ GPUs)
        torch.set_float32_matmul_precision('high')
        
        print('Using Device: ', config.device)

        
        self.data_path = os.path.abspath(os.path.expanduser(config.data_path))
        self.images_path = os.path.abspath(os.path.expanduser(config.images_path))
        self.dataframe_path = os.path.abspath(os.path.expanduser(config.dataframe_path))
        self.embedding_size = config.embedding_size
        self.results_path = os.path.abspath(os.path.expanduser(config.results_path))
        self.testing_data_path = os.path.abspath(os.path.expanduser(config.testing_data_path))

        
        os.makedirs(self.results_path, exist_ok=True)
        
        self.norm_method = config.norm_method # layer or batch
        self.random_seed = config.random_seed
        self.score_limit = config.score_limit
        self.field = config.field
        
        self.latents_scaled = config.latents_scaled
        
        self.maximum_ensemble_size = config.maximum_ensemble_size
        self.num_injections = config.num_injections
        
        self.supplement_ratio = config.supplement_ratio
        self.supplement_method = config.supplement_method # threshold or uncertainty
        self.num_submission_train = config.num_submission_train
        self.patience = config.patience
        self.warmup_epochs = config.warmup_epochs
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        
        self.fix_ensembles = config.fix_ensembles
        self.use_focal_loss = config.use_focal_loss
        self.cold_start = config.cold_start
        self.monitor_metric = config.monitor_metric
        self.checkpoint_min_epoch = getattr(config, 'checkpoint_min_epoch', 0) or 0
        
        self.device = config.device
        
        if self.use_focal_loss:
            print(f'Use focal loss with alpha = {config.focal_loss_alpha} and gamma = {config.focal_loss_gamma}')
            self.criterion = BinaryFocalLoss(alpha=config.focal_loss_alpha,
                                             gamma=config.focal_loss_gamma)
        else:
            print('Use BCE loss')
            self.criterion = nn.BCEWithLogitsLoss()
            
        self.ensembles = []
        self.current_round = 0
        self.selected_sl_names = []
        self.selected_non_sl_names = []
        self.testing_sl_names = []
        # self.selected_testing_sl_names = []
        self.selected_injected_names = []
        self.testing_non_sl_names = []
        self.total_submissions = 0  # Counter for total submissions
        self.sl_name_incremented = []
        self.selected_names_measured = []
        
        self.num_increment_history = []
        self.mean_increment = 0
        
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        self.available_names = []
        
        self.model_trained = False
        self.scoring_in_progress = False
        self.scoring_last_error = None
        self.scoring_job_seq = 0
        self.scoring_completed_seq = 0
        self.scores = {}
        self.ranks_injections = {}
        self.injected_recovery = {}
        self.recovered_ratio = 0.0
        self.recovered_purity = 0.0
        self.filtered_out_names = []
        self.round_save_path = None
        self.testing_rank_metrics = {}
        
        # Initialize stats dictionary to avoid AttributeError
        self.stats = {
            'min_score': 0.0,
            'min_score_cowls': 0.0,
            'num_over_min_score': 0,
            'num_over_min_score_cowls': 0,
            'num_lower_max_non_sl_score': 0,
        }
        
        self.initialize_ensembles()
        self.load_data()
        self.load_history()
        
    def load_data(self):
        
        print(f'Loading embeddings from {self.data_path}')
        data = np.load(self.data_path, allow_pickle=True)
        self.latents = data['embeddings'].astype(np.float32)
        
        print(f'Loading dataframe from {self.dataframe_path}')
        self.df = pd.read_csv(self.dataframe_path, low_memory=False)
        self.df_bk = self.df.copy()
        
        if self.field == 'web':
            print('Use WEB dataset')

            field_mask = self.df['field'].astype(str).str.upper() == 'WEB'
            overlap_mask = self.df['another_name'].astype(str).str.contains('Web', na=False)
            idx = field_mask | overlap_mask

            self.df = self.df[idx].reset_index(drop=True)
            
            overlap_selected_mask = self.df['another_name'].astype(str).str.contains('Web', na=False)
            self.df.loc[overlap_selected_mask, 'name'] = self.df.loc[overlap_selected_mask, 'another_name'].astype(str)
            
            print(f'WEB field sources: {int(field_mask.sum())}')
            print(f'WEB-overlap sources (another_name contains Web): {int(overlap_mask.sum())}')
            print(f'Total selected for WEB dataset: {int(idx.sum())}')
            
        elif self.field == 'spring':
            # containts spring source and web COWLS sources
            print('Use SPRING field')
            field_mask = self.df['field'] == 'SPR'
            self.df_spr = self.df[field_mask].reset_index(drop=True)
            overlap_selected_mask = self.df_spr['another_name'].astype(str).str.contains('COS', na=False)
            self.df_spr.loc[overlap_selected_mask, 'name'] = self.df_spr.loc[overlap_selected_mask, 'another_name'].astype(str)
            
            # num_web = sum('Web' in name for name in self.df_spr['name'].values.astype(str))
            # print('Number of WEB sources in SPRING: ', num_web)

            field_mask = self.df['field'].astype(str).str.upper() == 'WEB'
            overlap_mask = self.df['another_name'].astype(str).str.contains('Web', na=False)
            idx = field_mask | overlap_mask

            self.df_web = self.df[idx].reset_index(drop=True)
            
            overlap_selected_mask = self.df_web['another_name'].astype(str).str.contains('Web', na=False)
            self.df_web.loc[overlap_selected_mask, 'name'] = self.df_web.loc[overlap_selected_mask, 'another_name'].astype(str)
            
            df_cowls = self.df_web[(self.df_web['COWLS'] == 1) & (self.df_web['primary'] == 1)]
            df_non_cowls = self.df_spr[self.df_spr['COWLS'] == 0]
            
            print('Number of COWLS sources from WEB: ', len(df_cowls))
            print('Number of non-COWLS sources from SPIRNG: ', len(df_non_cowls))
            
            self.df = pd.concat([df_non_cowls, df_cowls]).reset_index(drop=True)
        
        else:
            raise ValueError(f'Invalid field: {self.field}')
        
        df_non_cowls = self.df[self.df['COWLS'] == 0]
        df_cowls = self.df[(self.df['COWLS'] == 1) & (self.df['primary'] == 1)]
        
        print(f'Number of non-COWLS sources: {len(df_non_cowls)}')
        print(f'Number of COWLS sources: {len(df_cowls)}')

        # reframed dataframe
        self.df = pd.concat([df_non_cowls, df_cowls]).reset_index(drop=True)
        print('Dataframe size: ', len(self.df))
        
        self.galaxy_names = self.df['name'].values.astype(str)
        
        indices = self.df['embedding_index'].values.astype(int)
        self.name_to_idx = {name: idx for name, idx in zip(self.galaxy_names, indices)}
        
        score_order = ['M25'] + [f'S{i:02d}' for i in range(0, 13)[::-1]]

        if self.score_limit in score_order:
            # filter COWLS sources
            
            selected_scores = score_order[:score_order.index(self.score_limit) + 1]
            print('Consider COWLS scores: ', selected_scores)
            cowls_names = self.df[(self.df['score'].isin(selected_scores)) & (self.df['COWLS'] == 1)]['name'].tolist()
            print(f'COWLS sources considered: {len(cowls_names)}')
            
            excluded_cowls_names = self.df[~self.df['score'].isin(selected_scores) & (self.df['COWLS'] == 1)]['name'].tolist()

        else:
            print(f'{self.score_limit} is not valid score limit, consider all COWLS sources')
            cowls_names = self.df[self.df['COWLS'] == 1]['name'].tolist()
            excluded_cowls_names = []

        
        self.cowls_sl_names = cowls_names
        self.cowls_sl_scores = self.df[(self.df['name'].isin(self.cowls_sl_names))]['score'].tolist()
        
        self.name_to_score = {name: score for name, score in
                              zip(self.cowls_sl_names, self.cowls_sl_scores)}
        
        self.excluded_cowls_names = excluded_cowls_names
        excluded_cowls_scores = self.df[(self.df['name'].isin(excluded_cowls_names))]['score'].tolist()
        
        # for the same negative samples
        np.random.seed(self.random_seed)
        sample_idx = np.random.choice(len(excluded_cowls_names), 10, replace=False)
        print(f'Excluded COWLS sources: {np.array(excluded_cowls_names)[sample_idx]}')
        print(f'Excluded COWLS names examples: {np.array(excluded_cowls_names)[sample_idx]}')
        print(f'Excluded COWLS scores examples: {np.array(excluded_cowls_scores)[sample_idx]}')
        
        if isinstance(self.num_injections, int) and self.num_injections > 0:
            print(f'Select {self.num_injections} M25 sources as injected sources')
            np.random.seed(self.random_seed)
            m25_names = np.array(self.cowls_sl_names)[np.array(self.cowls_sl_scores) == 'M25']
            self.injected_names = np.random.choice(
                m25_names, 
                self.num_injections, 
                replace=False
            ).tolist()
            
            injected_scores = [self.name_to_score[name] for name in self.injected_names]
            
            print(f'Injected names: {self.injected_names}')
            print(f'Injected scores: {injected_scores}')
            
            os.makedirs(os.path.join(self.results_path, 'injected'), exist_ok=True)
            for name in self.injected_names:
                shutil.copy(os.path.join(self.images_path, f'{name}.jpg'), 
                            os.path.join(self.results_path, 'injected', f'{name}.jpg'))
                
            print(f'Number of injected names: {len(self.injected_names)}')
            self.cowls_sl_names = np.setdiff1d(self.cowls_sl_names, self.injected_names).tolist()
        
        else:
            print('Use all M25 sources as injected sources')
            m25_names = np.array(self.cowls_sl_names)[np.array(self.cowls_sl_scores) == 'M25']
            self.injected_names = m25_names.tolist()
            injected_scores = [self.name_to_score[name] for name in self.injected_names]
            
            print(f'Injected names: {self.injected_names}')
            print(f'Injected scores: {injected_scores}')
            
            os.makedirs(os.path.join(self.results_path, 'injected'), exist_ok=True)
            for name in self.injected_names:
                shutil.copy(os.path.join(self.images_path, f'{name}.jpg'), 
                            os.path.join(self.results_path, 'injected', f'{name}.jpg'))
                
            print(f'Number of injected names: {len(self.injected_names)}')
            self.cowls_sl_names = np.setdiff1d(self.cowls_sl_names, self.injected_names).tolist()
            print('Number of COWLS sources after removing injected sources: ', len(self.cowls_sl_names))
        
        # M25 injected sources are fixed testing SL set.
        self.testing_sl_names = list(self.injected_names)
        
        if os.path.exists(self.testing_data_path):
            print(f'Testing data initialized from {self.testing_data_path}')
            with open(self.testing_data_path, 'r') as f:
                testing_data = json.load(f)
                
            self.testing_sl_names = testing_data['testing_sl_names']
            self.testing_non_sl_names = testing_data['testing_non_sl_names']
            
            self.current_round = 1
            
            count = 0
            for name in self.testing_sl_names + self.testing_non_sl_names:
                if name not in list(self.name_to_idx.keys()):
                    self.name_to_idx[name] = self.df_bk[self.df_bk['name'] == name]['embedding_index'].values[0]
                    count += 1
                    
            print('Number added to name_to_idx: ', count)
            
        print(f'Testing SL: {len(self.testing_sl_names)}')
        print(f'Testing non-SL: {len(self.testing_non_sl_names)}')

        # initialize SL pool
        self.selected_sl_names += self.cowls_sl_names
        
        self.last_sl_count = len(self.selected_sl_names)
        
        print(f'Initial selected SL names: {len(self.selected_sl_names)}')
        
        # move latents to cuda device
        self.latents = torch.from_numpy(self.latents).to(self.device, non_blocking=True)
        
        if self.latents_scaled:
            
            print('Scale latents using standard scaler')
            
            # Convert to numpy for StandardScaler, then back to torch tensor
            latents_np = self.latents.cpu().numpy()
            scaler = StandardScaler()
            latents_np = scaler.fit_transform(latents_np)
            self.latents = torch.from_numpy(latents_np.astype(np.float32)).to(config.device, non_blocking=True)
            
            with open(os.path.join(self.results_path, 'scaler.pkl'), 'wb') as f:
                joblib.dump(scaler, f)
    
    def initialize_ensembles(self):
        
        self.ensembles = []
        for i in range(self.maximum_ensemble_size):
            model = LatentClassifier(input_dim=self.embedding_size, d_ffn_factor=2, depth=2, 
                                 bayesian=False).to(config.device)
            self.ensembles.append(model)
        
        params, buffers = stack_module_state(self.ensembles)
        self.optimizer = optim.Adam(params.values(), lr=self.learning_rate, weight_decay=1e-5, 
                                    fused=False)
        self.scaler = torch.amp.GradScaler(device=config.device)
        
        def fmodel(params, buffers, x):
            return functional_call(self.ensembles[0], (params, buffers), x)
        
        self.predict_ensemble = vmap(fmodel, in_dims=(0, 0, None))
        self.params = params
        self.buffers = buffers
        
        self.ensemble_size = len(self.ensembles)
        
        print(f'Initialized {self.ensemble_size} ensembles')
    
    def add_ensemble(self):
        
        print('Adding ensemble...')
        
        model = LatentClassifier(input_dim=self.embedding_size, d_ffn_factor=2, depth=2, 
                                bayesian=False).to(config.device)
        self.ensembles.append(model)
        
        if len(self.ensembles) > self.maximum_ensemble_size:
            print(f'Ensemble size exceeded {self.maximum_ensemble_size}. Removing oldest model...')
            self.ensembles.pop(0)
            
        self.ensemble_size = len(self.ensembles)
        
        print('Current ensemble size: ', self.ensemble_size)
        
        params, buffers = stack_module_state(self.ensembles)
        self.optimizer = optim.Adam(params.values(), lr=self.learning_rate, weight_decay=1e-5, 
                                    fused=False)
        self.scaler = torch.amp.GradScaler(device=config.device)
        
        def fmodel(params, buffers, x):
            return functional_call(self.ensembles[0], (params, buffers), x)
        
        self.predict_ensemble = vmap(fmodel, in_dims=(0, 0, None))
        self.params = params
        self.buffers = buffers
    
    def get_available_galaxies(self):
        
        excluded_names = (
            self.selected_sl_names # user selected SL, training + testing
            + self.selected_non_sl_names # user selected SL, training
            + self.selected_injected_names # user selected injected sources
            + self.testing_non_sl_names # testing non-SL
        )
        if self.field == 'spring':
            excluded_names += self.excluded_cowls_names
        
        excluded_names = np.array(excluded_names)
        
        self.available_names = np.setdiff1d(self.galaxy_names, excluded_names)
        if self.field == 'spring':
            num_web = sum('Web' in name for name in self.available_names.astype(str))
            print('Number of WEB sources in available names: ', num_web)
        
        return len(self.available_names)
    
    def get_random_batch(self, size=10):
        
        if self.current_round == 0:
            excluded_names = (
                self.selected_sl_names 
                + self.selected_non_sl_names
                + self.selected_injected_names
                + self.testing_non_sl_names
                + self.excluded_cowls_names
            )
        else:
            excluded_names = (
                self.selected_sl_names 
                + self.selected_non_sl_names
                + self.selected_injected_names
                + self.testing_non_sl_names
            )
            
            if self.field == 'spring':
                excluded_names += self.excluded_cowls_names
        
        self.available_names = np.setdiff1d(self.galaxy_names, excluded_names)
        if self.field == 'spring':
            num_web = sum('Web' in name for name in self.available_names.astype(str))
            print('Number of WEB sources in available names: ', num_web)
        
        available_size = min(size, len(self.available_names))
        if available_size == 0:
            return [], []
        
        np.random.seed(self.random_seed) # fix first random batch
        selected_names = np.random.choice(self.available_names, available_size, replace=False)
        
        if self.model_trained and self.scores:
            selected_scores = [self.scores.get(name, 0.5) for name in selected_names]
        else:
            selected_scores = [0.5] * available_size
            
        return selected_names.tolist(), selected_scores
    
    def add_selections(self, sl_names, non_sl_names):
        
        round_save_path = os.path.join(self.results_path, f'round_{self.current_round}')
        os.makedirs(os.path.join(round_save_path, 'sl_selections'), exist_ok=True)
        
        for name in sl_names:
            self.selected_sl_names.append(name)
            
            # copy sl selection to round path
            shutil.copy(os.path.join(self.images_path, f'{name}.jpg'), 
                        os.path.join(round_save_path, 'sl_selections', f'{name}.jpg'))
                
        for name in non_sl_names:
            if self.current_round == 0:
                self.testing_non_sl_names.append(name)
            else:
                self.selected_non_sl_names.append(name)
        
        self.total_submissions += len(sl_names) + len(non_sl_names)
        
        print(f'Current selected SL count: {len(self.selected_sl_names)}')
        print(f'Current selected non-SL count: {len(self.selected_non_sl_names)}')
    
    def plot_history(self, train_loss_history, test_loss_history, lr_history, 
                     best_epoch, save_path, separation_history=None):
        
        fig, ax = plt.subplots(1, 2, figsize=(20, 6))
        
        # Loss curves
        if train_loss_history:
            ax[0].plot(train_loss_history, label='Train Loss', color='tab:blue')
        if test_loss_history:
            ax[0].plot(test_loss_history, label='Test Loss', color='tab:orange')
        ax[0].axvline(best_epoch, color='tab:red', linestyle='--', label='Best Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].grid(True, alpha=0.3)
        if train_loss_history or test_loss_history:
            ax[0].legend()
        
        # Right panel: test rank separation (when monitored) + LR on twin axis; else LR only.
        if separation_history is not None and len(separation_history) > 0:
            ax[1].plot(separation_history, label='Test rank separation', color='tab:purple')
            ax[1].set_ylabel('Median rank(non-SL) − median rank(SL)')
            ax[1].set_xlabel('Epoch')
            ax[1].set_title('Testing separation & learning rate')
            ax[1].axvline(best_epoch, color='tab:red', linestyle='--', label=f'Best Epoch: {best_epoch}')
            ax[1].grid(True, alpha=0.3)
            ax[1].legend(loc='upper left')
            if lr_history:
                ax_lr = ax[1].twinx()
                ax_lr.plot(lr_history, color='tab:green', alpha=0.8, label='Learning Rate')
                ax_lr.set_yscale('log')
                ax_lr.set_ylabel('Learning Rate')
                ax_lr.legend(loc='upper right')
        else:
            if lr_history:
                ax[1].plot(lr_history, label='Learning Rate', color='tab:green')
                ax[1].set_yscale('log')
                ax[1].legend()
                ax[1].axvline(best_epoch, color='tab:red', linestyle='--', label=f'Best Epoch: {best_epoch}')
            ax[1].set_ylabel('Learning Rate')
            ax[1].set_xlabel('Epoch')
            ax[1].set_title('Learning Rate Schedule')
            ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        out = os.path.join(save_path, 'history.png')
        plt.savefig(out)
        plt.close()
        print(f'Saved history plot to {out}')
    
    def compute_testing_rank_separation_from_preds(self, test_preds_logits, n_sl=None, n_ns=None):
        """
        Rank testing SL and testing non-SL sources by mean ensemble probability within the
        hold-out test pool only (not the full catalog). Rank 1 = highest score.

        Separation = median(rank among non-SL test sources) − median(rank among SL test sources).
        Larger separation ⇒ positives are ranked higher than negatives within the test pool.

        Args:
            test_preds_logits: tensor of shape (ensemble_size, n_test), aligned with
                testing_sl_names followed by testing_non_sl_names.
            n_sl, n_ns: optional counts when the logit tensor is built from a filtered subset
                (e.g. RGB paths missing for some testing names) but still SL block then non-SL block.

        Returns:
            dict with median_rank_testing_sl, median_rank_testing_non_sl, separation,
            n_testing_sl, n_testing_non_sl (separation / medians None if undefined).
        """
        n_sl = len(self.testing_sl_names) if n_sl is None else int(n_sl)
        n_ns = len(self.testing_non_sl_names) if n_ns is None else int(n_ns)
        if n_sl == 0 or n_ns == 0:
            return {
                'median_rank_testing_sl': None,
                'median_rank_testing_non_sl': None,
                'separation': None,
                'n_testing_sl': n_sl,
                'n_testing_non_sl': n_ns,
            }

        with torch.no_grad():
            probs = torch.sigmoid(test_preds_logits).mean(dim=0).float().cpu().numpy()

        # Average ranks for ties (method='average'); rank 1 = highest probability.
        ranks = rankdata(-probs, method='average')
        sl_ranks = ranks[:n_sl]
        ns_ranks = ranks[n_sl:]
        med_sl = float(np.median(sl_ranks))
        med_ns = float(np.median(ns_ranks))
        sep = med_ns - med_sl

        return {
            'median_rank_testing_sl': med_sl,
            'median_rank_testing_non_sl': med_ns,
            'separation': sep,
            'n_testing_sl': n_sl,
            'n_testing_non_sl': n_ns,
        }

    def train_embeddings(self):
        
        # Round 0 is reserved for collecting testing labels.
        if self.current_round == 0:
            print('Skip training in round 0: labels are used as testing set initialization.')
            # Advance to round 1 so subsequent training can run normally.
            self.current_round += 1
            # Round 1 starts a fresh submission counter.
            self.total_submissions = 0
            
            # if not os.path.exists(self.testing_data_path):
            
            #     with open(self.testing_data_path, 'w') as f:
            #         json.dump({
            #             'testing_sl_names': self.testing_sl_names,
            #             'testing_non_sl_names': self.testing_non_sl_names,
            #         }, f, indent=4)
            
            return {
                'success': True,
                'skipped': True,
                'reason': 'round_0_testing_only',
                'round': self.current_round,
            }
        
        increment = len(self.selected_sl_names) - self.last_sl_count
        self.num_increment_history.append(increment)
        self.last_sl_count = len(self.selected_sl_names)
        
        print('Increment SL sources: ', increment)
        
        if self.fix_ensembles == False and self.cold_start == False:
            print('Adding ensemble...')
            self.add_ensemble()
        
        if self.fix_ensembles == True and self.cold_start == True:
            print('Resetting ensembles...')
            self.initialize_ensembles()
        
        # Remove sources near injected sources and keep recovery history.
        (
            selected_sl_names_filtered,
            selected_non_sl_names_filtered,
        ) = self.filter_injected_sources(self.selected_sl_names, self.selected_non_sl_names)
        
        sl_count = len(selected_sl_names_filtered)
        non_sl_count = len(selected_non_sl_names_filtered)
        total_count = sl_count + non_sl_count
        
        print(f"\n=== Training ensembles - Round {self.current_round} ===")
        print(f"Total samples: {total_count} (SL: {sl_count}, Non-SL: {non_sl_count})")
        print(f"Patience: {self.patience}, Warmup: {self.warmup_epochs}")
        if self.checkpoint_min_epoch > 0:
            print(f"Checkpoint min epoch: {self.checkpoint_min_epoch} (best checkpoint only after this)")
        print("=" * 60)

        if self.monitor_metric == 'separation':
            if len(self.testing_sl_names) == 0 or len(self.testing_non_sl_names) == 0:
                raise ValueError(
                    'monitor_metric="separation" requires non-empty testing_sl_names '
                    'and testing_non_sl_names.'
                )
        
        training_names = np.array(selected_sl_names_filtered + selected_non_sl_names_filtered)
        training_labels = np.array([1] * len(selected_sl_names_filtered) + [0] * len(selected_non_sl_names_filtered))
        training_labels = torch.from_numpy(training_labels).to(self.device, non_blocking=True).float()
        training_indices = np.array([self.name_to_idx[name] for name in training_names.tolist()])
        training_indices = torch.from_numpy(training_indices).to(self.device, non_blocking=True)
        training_latents = self.latents[training_indices]
        
        testing_names = np.array(self.testing_sl_names + self.testing_non_sl_names)
        testing_labels = np.array([1] * len(self.testing_sl_names) + [0] * len(self.testing_non_sl_names))
        testing_labels = torch.from_numpy(testing_labels).to(self.device, non_blocking=True).float()
        testing_indices = np.array([self.name_to_idx[name] for name in testing_names.tolist()])
        testing_indices = torch.from_numpy(testing_indices).to(self.device, non_blocking=True)
        testing_latents = self.latents[testing_indices]

        sample_weights = None
        if not self.use_focal_loss:
            labels, counts = torch.unique(training_labels, return_counts=True)
            class_weights = 1.0 / counts.float()
            class_sample_weights = class_weights[training_labels.long()]
            sample_weights = class_sample_weights
            print('Use weighted sampling with BCE loss')
            print('Labels: ', labels.cpu().numpy())
            print('Counts: ', counts.cpu().numpy())
            print('Class weights: ', class_weights.cpu().numpy())
            print(
                f'Class sample weights - min: {class_sample_weights.min():.4f}, '
                f'max: {class_sample_weights.max():.4f}, '
                f'mean: {class_sample_weights.mean():.4f}'
            )

        dataloader = CudaDataLoader(training_latents, training_labels, self.batch_size, sample_weights)
        num_batches = len(dataloader)
        
        # Ensure each round starts from the configured base LR.
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate
        
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max' if self.monitor_metric == 'separation' else 'min',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )
        
        train_loss_history = []
        test_loss_history = []
        lr_history = []
        separation_history = []
        best_test_loss = float('inf')
        best_train_loss = float('inf')
        best_separation = float('-inf')
        best_separation_metrics = None
        best_epoch = -1
        best_params = copy.deepcopy(self.params)
        patience_counter = 0
        
        start_time = time.time()
        epoch = 0
        while True:
            train_epoch_loss = torch.tensor(0.0, device=self.device)
            for batch_latents, batch_labels in dataloader:
                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    preds = self.predict_ensemble(self.params, self.buffers, batch_latents)
                    loss = self.criterion(preds.reshape(-1, 1), batch_labels.expand(self.ensemble_size, -1).reshape(-1, 1))
                    train_epoch_loss += loss.detach()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            train_loss = (train_epoch_loss / num_batches).item()
            train_loss_history.append(train_loss)
            
            with torch.no_grad():
                test_preds = self.predict_ensemble(self.params, self.buffers, testing_latents)
                test_loss = self.criterion(
                    test_preds.reshape(-1, 1),
                    testing_labels.expand(self.ensemble_size, -1).reshape(-1, 1)
                ).item()
                test_loss_history.append(test_loss)

            sep_metrics = None
            if self.monitor_metric == 'separation':
                sep_metrics = self.compute_testing_rank_separation_from_preds(test_preds)
                s = sep_metrics['separation']
                separation_history.append(float(s) if s is not None else float('nan'))

            if epoch % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                line = (
                    f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
                )
                if self.monitor_metric == 'separation' and sep_metrics is not None:
                    if sep_metrics['separation'] is not None:
                        line += (
                            f" | Sep: {sep_metrics['separation']:.3f} "
                            f"(med rank SL {sep_metrics['median_rank_testing_sl']:.2f}, "
                            f"non-SL {sep_metrics['median_rank_testing_non_sl']:.2f})"
                        )
                print(line)
                print(f"Current LR: {current_lr:.2e}")

            # Keep warmup epochs for stabilization; start model selection after warmup using test loss.
            
            if self.monitor_metric == 'test_loss':
            
                lr_scheduler.step(test_loss)
            
                if epoch >= self.warmup_epochs and epoch >= self.checkpoint_min_epoch:
                    if test_loss < best_test_loss:
                        best_test_loss = test_loss
                        best_params = copy.deepcopy(self.params)
                        best_epoch = epoch
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            print(f"Early stopping at epoch {epoch} (Best Test Loss: {best_test_loss:.4f})")
                            break
            
            elif self.monitor_metric == 'train_loss':
                
                lr_scheduler.step(train_loss)
                
                if epoch >= self.warmup_epochs and epoch >= self.checkpoint_min_epoch:
                    if train_loss < best_train_loss:
                        best_train_loss = train_loss
                        best_params = copy.deepcopy(self.params)
                        best_epoch = epoch
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            print(f"Early stopping at epoch {epoch} (Best Train Loss: {best_train_loss:.4f})")
                            break
            
            elif self.monitor_metric == 'separation':
                sep = sep_metrics['separation'] if sep_metrics else None
                if sep is not None:
                    lr_scheduler.step(sep)
                    if epoch >= self.warmup_epochs and epoch >= self.checkpoint_min_epoch:
                        if sep > best_separation:
                            best_separation = sep
                            best_params = copy.deepcopy(self.params)
                            best_epoch = epoch
                            patience_counter = 0
                            best_separation_metrics = {
                                'median_rank_testing_sl': sep_metrics['median_rank_testing_sl'],
                                'median_rank_testing_non_sl': sep_metrics['median_rank_testing_non_sl'],
                                'separation': sep,
                                'n_testing_sl': sep_metrics['n_testing_sl'],
                                'n_testing_non_sl': sep_metrics['n_testing_non_sl'],
                            }
                        else:
                            patience_counter += 1
                            if patience_counter >= self.patience:
                                print(
                                    f"Early stopping at epoch {epoch} "
                                    f"(Best test separation: {best_separation:.4f})"
                                )
                                break
                
            lr_history.append(self.optimizer.param_groups[0]['lr'])
            epoch += 1
        
        end_time = time.time()
        training_time = end_time - start_time
        print(f'Training time: {training_time:.2f} seconds')
        
        self.best_params = best_params
        self.model_trained = True
        
        self.round_save_path = os.path.join(self.results_path, f'round_{self.current_round}')
        os.makedirs(self.round_save_path, exist_ok=True)
        
        torch.save({'ensemble_params': self.best_params, 'buffers': self.buffers, 'config': {'num_ensembles': self.ensemble_size}}, 
                   os.path.join(self.round_save_path, 'model.pth'))
        
        hist_payload = {
            'train_loss_history': train_loss_history,
            'test_loss_history': test_loss_history,
            'lr_history': lr_history,
        }
        if self.monitor_metric == 'separation':
            hist_payload['separation_history'] = separation_history
        with open(os.path.join(self.round_save_path, 'history.json'), 'w') as f:
            json.dump(hist_payload, f, indent=4)
        
        sep_hist = separation_history if self.monitor_metric == 'separation' else None
        self.plot_history(
            train_loss_history,
            test_loss_history,
            lr_history,
            best_epoch,
            self.round_save_path,
            separation_history=sep_hist,
        )
        
        print(f'Best epoch: {best_epoch}')
        if self.monitor_metric == 'test_loss':
            print(f'Best testing loss: {best_test_loss:.4f}')
            self.testing_rank_metrics = {}
        elif self.monitor_metric == 'train_loss':
            print(f'Best training loss: {best_train_loss:.4f}')
            self.testing_rank_metrics = {}
        elif self.monitor_metric == 'separation':
            self.testing_rank_metrics = best_separation_metrics if best_separation_metrics else {}
            if best_separation_metrics:
                print(
                    f"Best test rank separation: {best_separation_metrics['separation']:.4f} "
                    f"(median rank SL {best_separation_metrics['median_rank_testing_sl']:.2f}, "
                    f"non-SL {best_separation_metrics['median_rank_testing_non_sl']:.2f})"
                )
            else:
                print('Best test rank separation: (no improving checkpoint recorded)')

        self.update_scores()
        os.makedirs(os.path.join(self.round_save_path, 
                                 'sl_selections'), exist_ok=True)
        
        # Rank by score (high to low): rank 1 means highest score.
        scores_sorted = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        score_ranks = {name: rank + 1 for rank, (name, _) in enumerate(scores_sorted)}
        ranks_injections_dict = {name: score_ranks.get(name) for name in self.injected_names}
        valid_ranks = [rank for rank in ranks_injections_dict.values() if rank is not None]

        mean_ranks = float(np.mean(valid_ranks)) if valid_ranks else None
        median_ranks = float(np.median(valid_ranks)) if valid_ranks else None

        self.ranks_injections = {
            'mean_rank': mean_ranks,
            'median_rank': median_ranks,
            'ranks_injections': ranks_injections_dict,
        }

        if mean_ranks is not None:
            print(f'Mean rank of injected sources: {mean_ranks:.2f}')
            print(f'Median rank of injected sources: {median_ranks:.2f}')
        else:
            print('No injected source ranks available.')
        
        recovered_names = self.injected_recovery.keys()
        print(f'Injected recovered names: {recovered_names}')
        print(f'Number of injected recovered names: {len(recovered_names)}')
        
        self.recovered_ratio = len(recovered_names) / len(self.injected_names)
        self.recovered_purity = len(recovered_names) / (len(self.selected_sl_names) + len(self.selected_non_sl_names))
        
        self.create_visualizations()
        self.save_records()
        
        self.current_round += 1
        
        return {'success': True}
    
    def filter_injected_sources(self, selected_sl_names, selected_non_sl_names):
        
        print('Filtering selected sources near injected sources...')
        
        # Build coordinate lookup indexed by source name.
        coords_df = self.df[['name', 'ra', 'dec']].copy()
        coords_df['name'] = coords_df['name'].astype(str)
        coords_df = coords_df.drop_duplicates(subset='name').set_index('name')
        
        ra_injected = np.array([float(coords_df.at[name, 'ra']) for name in self.injected_names])
        dec_injected = np.array([float(coords_df.at[name, 'dec']) for name in self.injected_names])
        
        print(f'Injected sources RA: {np.around(ra_injected, 4)}')
        print(f'Injected sources Dec: {np.around(dec_injected, 4)}')
        
        inj_coords = SkyCoord(
            ra=ra_injected * u.deg,
            dec=dec_injected * u.deg,
            frame='icrs',
        )
        
        selected_names = selected_sl_names + selected_non_sl_names
        flags = [1] * len(selected_sl_names) + [0] * len(selected_non_sl_names)
        
        for source_name, flag in zip(selected_names, flags):
            
            if source_name in self.selected_names_measured:
                continue
            self.selected_names_measured.append(source_name)
            
            src_coord = SkyCoord(
                ra=float(coords_df.at[source_name, 'ra']) * u.deg,
                dec=float(coords_df.at[source_name, 'dec']) * u.deg,
                frame='icrs',
            )
            idx, d2d, _ = src_coord.match_to_catalog_sky(inj_coords)
            sep_constraint = d2d < 2.0 * u.arcsec
            if not sep_constraint:
                continue
            
            matched_injected_name = self.injected_names[idx]
            sep_arcsec = d2d.to(u.arcsec).value
            
            self.filtered_out_names.append(source_name)
            
            if flag == 1: # only record SL sources
            
                if matched_injected_name in self.injected_recovery.keys():
                    self.injected_recovery[matched_injected_name]['matched_sl_name'].append(source_name)
                    self.injected_recovery[matched_injected_name]['distance'].append(sep_arcsec)
                else:
                    self.injected_recovery[matched_injected_name] = {}
                    self.injected_recovery[matched_injected_name]['matched_sl_name'] = [source_name]
                    self.injected_recovery[matched_injected_name]['distance'] = [sep_arcsec]
        
        selected_sl_names_filtered = [name for name in selected_sl_names if name not in self.filtered_out_names]
        selected_non_sl_names_filtered = [name for name in selected_non_sl_names if name not in self.filtered_out_names]
            
        print(f'Filtered out names: {self.filtered_out_names}')
        print(f'Number of filtered out names: {len(self.filtered_out_names)}')
            
        return selected_sl_names_filtered, selected_non_sl_names_filtered
        
    def update_scores(self):
        
        print('Updating probs...')
        
        start_time = time.time()
        
        if not self.model_trained:
            self.scores = {}
            return

        all_names = self.galaxy_names
        # if self.field == 'spring':
        #     all_names = all_names.tolist() if isinstance(all_names, np.ndarray) else all_names
        #     injected_names = self.injected_names.tolist() if isinstance(self.injected_names, np.ndarray) else self.injected_names
        #     all_names += injected_names
        
        all_idx = np.array([self.name_to_idx[name] for name in all_names])
        all_idx = torch.from_numpy(all_idx).to(self.device, non_blocking=True)
        all_latents = self.latents[all_idx]
        
        dataloader = CudaDataLoader(
            all_latents,
            y_tensor=None,  # Prediction mode - only x_tensor needed
            batch_size=self.batch_size,
        )
        
        # Accumulate predictions on GPU and move to CPU once at the end
        scores_gpu = []
        for batch_latents in dataloader:
            
            with torch.no_grad():
                
                with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                    preds_logits = self.predict_ensemble(self.best_params, self.buffers, batch_latents)
                    # Convert logits to probabilities for scoring
                    preds = torch.sigmoid(preds_logits)
                    
            scores_gpu.append(preds)  # (num_ensembles, batch_size)
            
        end_time = time.time()
        updating_time = end_time - start_time
        print(f'Updating scores time: {updating_time:.2f} seconds')
                
        scores = torch.cat(scores_gpu, dim=1).cpu().numpy()  # (num_ensembles, num_samples)
        mean_scores = np.mean(scores, axis=0)
        uncertainties = np.std(scores, axis=0)
        
        self.scores = {name: score for name, score in zip(all_names, mean_scores)}
        self.uncertainties = {name: uncertainty for name, uncertainty in zip(all_names, uncertainties)}
        
        results = {}
        results['name'] = list(self.scores.keys())
        results['prob'] = list(self.scores.values())
        results['uncertainty'] = list(self.uncertainties.values())
        
        df = pd.DataFrame(results)
        df['selected_sl'] = df['name'].isin(self.selected_sl_names).astype(int)
        df['selected_non_sl'] = df['name'].isin(self.selected_non_sl_names).astype(int)
        df['selected_injected'] = df['name'].isin(self.selected_injected_names).astype(int)
        
        cowls_score_map = dict(self.name_to_score)
        df['COWLS'] = df['name'].isin(cowls_score_map).astype(int)
        # Map by galaxy name; non-COWLS rows get NaN
        df['COWLS_score'] = df['name'].map(cowls_score_map)
        
        df.sort_values(by=['COWLS', 'prob'], ascending=[False, False], inplace=True)
        df.to_csv(os.path.join(self.round_save_path, 'probs.csv'), index=False)
        
        print(f'Updated probs for {len(self.scores)} galaxies')
        
    def custom_serializer(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Type {type(obj)} not serializable')
    
    def save_records(self):
        print('Saving records...')
        records = {
            'round': self.current_round,
            'sl_count': len(self.selected_sl_names),
            'non_sl_count': len(self.selected_non_sl_names),
            'total_submissions': self.total_submissions,
            'mean_increment_previous_5_rounds': np.mean(self.num_increment_history[-5:]).item() if len(self.num_increment_history) > 0 else 0.0,
            'injected_recovery': self.injected_recovery,
            'recovered_ratio': self.recovered_ratio,
            'recovered_purity': self.recovered_purity,
            'ranks_injections': self.ranks_injections,
            'model_trained': self.model_trained,
            'model_resetted': self.model_resetted if hasattr(self, 'model_resetted') else False,
            'num_ensembles': self.ensemble_size,
            'min_score': self.stats['min_score'],
            'num_over_min_score': self.stats['num_over_min_score'],
            'min_score_cowls': self.stats['min_score_cowls'],
            'num_over_min_score_cowls': self.stats['num_over_min_score_cowls'],
            'num_lower_max_non_sl_score': self.stats.get('num_lower_max_non_sl_score', 0),
            'num_increment_history': list(self.num_increment_history),
            'sl_names': self.selected_sl_names,
            'non_sl_names': self.selected_non_sl_names,
            'testing_sl_names': self.testing_sl_names,
            'testing_non_sl_names': self.testing_non_sl_names,
            'selected_names_measured': self.selected_names_measured,
            'filtered_out_names': self.filtered_out_names,
            'testing_rank_metrics': getattr(self, 'testing_rank_metrics', {}),
        }
        
        with open(os.path.join(self.round_save_path, 'records.json'), 'w') as f:
            json.dump(records, f, indent=4, default=self.custom_serializer)
            
        self.plot_selection_history()
        self.plot_median_rank_history()
        
    def plot_median_rank_history(self):
        rounds = []
        median_ranks = []

        if not os.path.exists(self.results_path):
            print(f'Results path does not exist: {self.results_path}')
            return

        round_dirs = []
        for dirname in os.listdir(self.results_path):
            if not dirname.startswith('round_'):
                continue
            try:
                round_num = int(dirname.split('_')[1])
            except (IndexError, ValueError):
                continue
            round_dirs.append((round_num, dirname))

        round_dirs.sort(key=lambda x: x[0])

        for round_num, dirname in round_dirs:
            records_path = os.path.join(self.results_path, dirname, 'records.json')
            if not os.path.exists(records_path):
                continue
            try:
                with open(records_path, 'r') as f:
                    records = json.load(f)
            except Exception as e:
                print(f'Failed reading {records_path}: {e}')
                continue

            ranks_injections = records.get('ranks_injections') or {}
            median_rank = ranks_injections.get('median_rank')
            if median_rank is None:
                continue

            rounds.append(round_num)
            median_ranks.append(float(median_rank))

        if len(rounds) == 0:
            print('No round records with ranks_injections median_rank found.')
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            rounds,
            median_ranks,
            marker='o',
            linewidth=2,
            color='tab:green',
            label='Median rank (injected sources)',
        )
        ax.set_xlabel('Round')
        ax.set_ylabel('Median Rank (1 = highest score)')
        ax.set_title('Median Injection Rank vs Round')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        save_path = os.path.join(self.results_path, 'median_rank_history.png')
        plt.savefig(save_path)
        plt.close()
        print(f'Saved median rank history plot: {save_path}')

    def plot_selection_history(self):
        rounds = []
        mean_increments = []

        if not os.path.exists(self.results_path):
            print(f'Results path does not exist: {self.results_path}')
            return

        round_dirs = []
        for dirname in os.listdir(self.results_path):
            if not dirname.startswith('round_'):
                continue
            try:
                round_num = int(dirname.split('_')[1])
            except (IndexError, ValueError):
                continue
            round_dirs.append((round_num, dirname))

        round_dirs.sort(key=lambda x: x[0])

        for round_num, dirname in round_dirs:
            records_path = os.path.join(self.results_path, dirname, 'records.json')
            if not os.path.exists(records_path):
                continue
            try:
                with open(records_path, 'r') as f:
                    records = json.load(f)
            except Exception as e:
                print(f'Failed reading {records_path}: {e}')
                continue

            metric = records.get('mean_increment_previous_5_rounds', None)
            if metric is None:
                continue

            rounds.append(round_num)
            mean_increments.append(float(metric))

        if len(rounds) == 0:
            print('No round records with mean_increment_previous_5_rounds found.')
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            rounds,
            mean_increments,
            marker='o',
            linewidth=2,
            color='tab:blue',
            label='Mean increment (previous 5 rounds)',
        )
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='y = 1.0')
        ax.set_xlabel('Round')
        ax.set_ylabel('Mean Increment')
        ax.set_title('Mean Increment vs Round')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        save_path = os.path.join(self.results_path, 'selection_history.png')
        plt.savefig(save_path)
        plt.close()
        print(f'Saved selection history plot: {save_path}')
        
    def create_visualizations(self):
        
        print('Creating visualizations...')
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        
        all_scores = list(self.scores.values())
        all_scores = np.array(all_scores)
        max_score = np.max(all_scores)
        
        thresholds = np.linspace(0, 1, 100)
        counts = np.array([np.sum(all_scores >= t) for t in thresholds])
        
        interp = interp1d(thresholds, counts, kind='cubic')
        
        def _prob(n):
            return float(self.scores.get(str(n), 0.5))

        # COWLS
        scores_for_cowls_sl = [_prob(name) for name in self.cowls_sl_names]
        interp_counts_cowls_sl = [interp(score) for score in scores_for_cowls_sl]
        
        sl_names_exclude_cowls = [name for name in self.selected_sl_names if name not in self.cowls_sl_names]
        scores_for_sl_exclude_cowls = [_prob(name) for name in sl_names_exclude_cowls]
        interp_counts_sl_exclude_cowls = [interp(score) for score in scores_for_sl_exclude_cowls]
        
        scores_for_non_sl = [_prob(name) for name in self.selected_non_sl_names]
        max_score_non_sl = float(np.max(scores_for_non_sl)) if len(scores_for_non_sl) > 0 else 0.0
        
        # min score - handle empty lists
        min_scores = []
        if len(scores_for_cowls_sl) > 0:
            min_scores.append(np.min(scores_for_cowls_sl))
        if len(scores_for_sl_exclude_cowls) > 0:
            min_scores.append(np.min(scores_for_sl_exclude_cowls))
            
        min_score = np.min(min_scores) if len(min_scores) > 0 else 0.0
        min_score_cowls = np.min(scores_for_cowls_sl) if len(scores_for_cowls_sl) > 0 else 0.0
        
        num_over_min_score = np.sum(all_scores > min_score)
        num_over_min_score_cowls = np.sum(all_scores > min_score_cowls)

        num_lower_max_non_sl_score = np.sum(all_scores < max_score_non_sl)
        
        self.stats = {
            'min_score': min_score,
            'min_score_cowls': min_score_cowls,
            'num_over_min_score': num_over_min_score,
            'num_over_min_score_cowls': num_over_min_score_cowls,
            'num_lower_max_non_sl_score': num_lower_max_non_sl_score,
        }
        
        print()
        print('stats: ')
        for key, value in self.stats.items():
            print(f'{key}: {value}')
        print()
        
        bins = np.arange(0, 1.02, 0.02)
        # hist, bin_edges = np.histogram(all_scores, bins=bins)
        # percentages = (hist / len(all_scores))
        # x_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        sl_scores = [_prob(name) for name in self.selected_sl_names]
        non_sl_scores = [_prob(name) for name in self.selected_non_sl_names]
        
        if len(sl_scores) > 0:
            print(f'Max SL score: {np.max(sl_scores)}')
            print(f'Min SL score: {np.min(sl_scores)}')
            print(f'Mean SL score: {np.mean(sl_scores)}')
        else:
            print('No SL scores for histogram (empty pool).')
        
        if len(non_sl_scores) > 0:
            print(f'Max non-SL score: {np.max(non_sl_scores)}')
            print(f'Min non-SL score: {np.min(non_sl_scores)}')
            print(f'Mean non-SL score: {np.mean(non_sl_scores)}')
        else:
            print('No non-SL scores for histogram (empty pool).')
        
        hist_sl, bin_edges_sl = np.histogram(sl_scores, bins=bins) if len(sl_scores) else (np.zeros(len(bins) - 1), bins)
        percentages_sl = (hist_sl / len(sl_scores)) if len(sl_scores) else np.zeros_like(hist_sl, dtype=float)
        x_bins_sl = (bin_edges_sl[:-1] + bin_edges_sl[1:]) / 2
        
        hist_non_sl, bin_edges_non_sl = np.histogram(non_sl_scores, bins=bins) if len(non_sl_scores) else (np.zeros(len(bins) - 1), bins)
        percentages_non_sl = (hist_non_sl / len(non_sl_scores)) if len(non_sl_scores) else np.zeros_like(hist_non_sl, dtype=float)
        x_bins_non_sl = (bin_edges_non_sl[:-1] + bin_edges_non_sl[1:]) / 2
        
        axes[0].bar(x_bins_sl, percentages_sl, width=0.02, alpha=0.7, color='skyblue', edgecolor='black',
                    label='SL')
        axes[0].bar(x_bins_non_sl, percentages_non_sl, width=0.02, alpha=0.7, color='orange', edgecolor='black',
                    label='Non-SL')
        axes[0].axvline(max_score, color='k', linestyle=':', linewidth=2,
                        label=f'Max Score: {max_score:.3f}')
        axes[0].axvline(min_score, color='purple', linestyle='-.', linewidth=2,
                        label=f'Min Score: {min_score:.3f}')
        axes[0].axvline(min_score_cowls, color='orange', linestyle=':', linewidth=2,
                        label=f'Min score (COWLS): {min_score_cowls:.3f}')
        axes[0].set_xlabel('Scores')
        axes[0].set_ylabel('Percentage')
        axes[0].set_title(f'Score Distribution - Round {self.current_round}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(counts, thresholds, 'b-', linewidth=2)
        
        axes[1].axhline(min_score, color='purple', linestyle='-.', linewidth=2,
                         label=f'Min score (SL): {min_score:.3f} ({num_over_min_score})')
        axes[1].axhline(min_score_cowls, color='orange', linestyle=':', linewidth=2, 
                        label=f'Min score (COWLS): {min_score_cowls:.3f} ({num_over_min_score_cowls})')
        axes[1].axhline(max_score_non_sl, color='green', linestyle=':', linewidth=2,
                        label=f'Max score (non-SL): {max_score_non_sl:.3f} ({num_lower_max_non_sl_score})')
        axes[1].scatter(interp_counts_cowls_sl, scores_for_cowls_sl, color='red', marker='o', s=30,
                        label=f'COWLS candidates ({len(self.cowls_sl_names)})')
        axes[1].scatter(interp_counts_sl_exclude_cowls, scores_for_sl_exclude_cowls, color='purple', marker='*', s=30,
                        label=f'Selected SL candidates ({len(sl_names_exclude_cowls)})')

        axes[1].set_xlabel('Number of samples')
        axes[1].set_ylabel('Scores')
        axes[1].set_title(f'Scores vs Number of Samples - Round {self.current_round}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        out = os.path.join(self.round_save_path, 'visualizations.png')
        plt.savefig(out)
        plt.close()
        
        print(f'Visualizations created: {out}')

        # `get_images` (supplement_method == 'threshold') uses this; keep in sync with viz stats.
        if len(scores_for_non_sl) > 0 and len(min_scores) > 0:
            self.dividing_threshold = float((min_score + max_score_non_sl) / 2.0)
        elif len(min_scores) > 0:
            self.dividing_threshold = float(min_score)
        else:
            self.dividing_threshold = 0.5
        
    def get_images(self):
        
        self.get_available_galaxies()
        if len(self.available_names) == 0:
            print('No available images')
            return [], []

        def _key(n):
            return str(n)

        def _score(n):
            k = _key(n)
            if k not in self.scores:
                print(f'get_images: missing score for {k!r}, using 0.5')
                return 0.5
            return float(self.scores[k])

        def _unc(n):
            k = _key(n)
            if k not in self.uncertainties:
                return 0.0
            return float(self.uncertainties[k])
        
        available_names = np.asarray(self.available_names)
        available_scores = np.array([_score(name) for name in available_names])
        available_uncertainties = np.array([_unc(name) for name in available_names])
        
        idx = np.argsort(available_scores)[::-1] # from high to low
        sorted_names = available_names[idx]

        n_avail = len(sorted_names)
        num_images = min(10, n_avail)
        if num_images <= 0:
            return [], []
        
        if self.supplement_ratio > 0:
            print('Get some supplement images by ratio: ', self.supplement_ratio)
            
            num_supplement = min(int(num_images * self.supplement_ratio), max(0, num_images - 1))
            num_high_score = num_images - num_supplement
            
            print('Number high score: ', num_high_score)
            print('Number supplement: ', num_supplement)
            
            high_names = sorted_names[:num_high_score]
            high_scores = np.array([_score(name) for name in high_names])
        
            print('High score names: ', high_names)
            print('High scores: ', high_scores)
            
            remaining_names = sorted_names[num_high_score:]
            remaining_scores = np.array([_score(name) for name in remaining_names])
            remaining_uncertainties = np.array([_unc(name) for name in remaining_names])

            div_thr = getattr(self, 'dividing_threshold', None)
            if div_thr is None:
                div_thr = float(np.median(remaining_scores)) if len(remaining_scores) else 0.5
            
            if self.supplement_method == 'threshold':
            
                print('Supplement by threshold')
                if len(remaining_scores) == 0:
                    supplement_names = np.array([], dtype=object)
                    supplement_scores = np.array([], dtype=np.float64)
                else:
                    idx = np.argsort(np.abs(remaining_scores - div_thr))
                    remaining_names = remaining_names[idx]
                    remaining_scores = remaining_scores[idx]
                    take = min(num_supplement, len(remaining_names))
                    supplement_names = remaining_names[:take]
                    supplement_scores = remaining_scores[:take]
                
                print('Supplement names: ', supplement_names)
                print('Supplement scores: ', supplement_scores)
            
            elif self.supplement_method == 'uncertainty':
                
                print('Supplement by uncertainty')
                if len(remaining_uncertainties) == 0:
                    supplement_names = np.array([], dtype=object)
                    supplement_scores = np.array([], dtype=np.float64)
                    supplement_uncertainties = np.array([], dtype=np.float64)
                else:
                    idx = np.argsort(remaining_uncertainties)[::-1] # from high to low
                    take = min(num_supplement, len(remaining_names))
                    supplement_names = remaining_names[idx][:take]
                    supplement_scores = remaining_scores[idx][:take]
                    supplement_uncertainties = remaining_uncertainties[idx][:take]
                
                print('Supplement names: ', supplement_names)
                print('Supplement scores: ', supplement_scores)
                print('Supplement uncertainties: ', supplement_uncertainties)
            
            selected_names = np.concatenate([high_names, supplement_names]) if len(supplement_names) else high_names
            selected_scores = np.concatenate([high_scores, supplement_scores]) if len(supplement_scores) else high_scores

        else:
            selected_names = sorted_names[:num_images]
            selected_scores = np.array([_score(name) for name in selected_names])
        
        names_out = [str(x) for x in np.asarray(selected_names).reshape(-1)]
        scores_out = [float(x) for x in np.asarray(selected_scores).reshape(-1)]
        return names_out, scores_out
        

    def load_history(self):
        
        files = ['history.json', 'history.png', 
                 'model.pth', 'records.json', 'probs.csv', 'visualizations.png']
        
        if config.checkpoint_round is None:
            
            print('No checkpoint round specified. Start from fresh.')       
             
        else:
            round_path = os.path.join(self.results_path, f'round_{config.checkpoint_round}')
            if not os.path.exists(round_path):
                raise Exception(f'Round {config.checkpoint_round} not found.')
                
            print(f'Loading checkpoint at {round_path}.')
            
            for file in files:
                file_path = os.path.join(round_path, file)
                if not os.path.exists(file_path):
                    round_num = config.checkpoint_round
                    raise Exception(f'File {file} not found in round {round_num}.')
            
            with open(os.path.join(round_path, 'records.json'), 'r') as f:
                records = json.load(f)
            
            self.current_round = records['round']
            self.selected_sl_names = list(records['sl_names'])
            self.selected_non_sl_names = list(records['non_sl_names'])
            self.testing_sl_names = list(records.get('testing_sl_names', []))
            self.selected_testing_sl_names = list(records.get('selected_testing_sl_names', []))
            self.testing_non_sl_names = list(records.get('testing_non_sl_names', []))
            self.total_submissions = records.get('total_submissions', 0) or 0
            self.model_trained = records['model_trained']
            self.round_save_path = round_path
            self.scoring_in_progress = False
            self.scoring_last_error = None
            self.scoring_job_seq = 0
            self.scoring_completed_seq = 0
            
            self.num_increment_history = records.get('num_increment_history', [])
            self.mean_increment = records.get('mean_increment_previous_5_rounds', 0.0)
            
            self.last_sl_count = len(self.selected_sl_names)
            
            self.sample_round_added = dict(records.get('sample_round_added', {}))
            
            self.stats['min_score'] = records['min_score']
            self.stats['min_score_cowls'] = records['min_score_cowls']
            self.stats['num_over_min_score'] = records['num_over_min_score']
            self.stats['num_over_min_score_cowls'] = records['num_over_min_score_cowls']
            self.stats['num_lower_max_non_sl_score'] = records.get('num_lower_max_non_sl_score', 0)
            
            # self.hold_out_true_rank_result = records.get('hold_out_rank_result')
            self.recovered_ratio = records.get('recovered_ratio', 0.0)
            self.recovered_purity = records.get('recovered_purity', 0.0)
            self.selected_names_measured = records.get('selected_names_measured', [])
            self.filtered_out_names = records.get('filtered_out_names', [])
            
            self.ranks_injections = records.get('ranks_injections', None)
            
            # Match post-train_ensembles() state: next round compares against full selection lists
            self.new_sl_names = list(self.selected_sl_names)
            self.new_non_sl_names = list(self.selected_non_sl_names)
            
            weights_path = os.path.join(round_path, 'model.pth')
            checkpoint = torch.load(weights_path, map_location=config.device)

            if checkpoint.get('kind') in {'resnet18_hitl', 'rgb_hitl'}:
                raise ValueError(
                    'This checkpoint is from the removed RGB/image training path. '
                    'Resume with an embedding-ensemble checkpoint (no "kind" field), '
                    'or retrain with the current embeddings-only detector.'
                )

            params = checkpoint['ensemble_params']
            buffers = checkpoint['buffers']
            model_config = checkpoint['config']

            self.params = params
            self.buffers = buffers
            self.ensemble_size = model_config['num_ensembles']
            self.best_params = copy.deepcopy(self.params)

            # Stack params must align with vmap batch size; rebuild if init used a different count
            if len(self.ensembles) != self.ensemble_size:
                self.ensembles = []
                for i in range(self.ensemble_size):
                    model = LatentClassifier(
                        input_dim=self.embedding_size,
                        d_ffn_factor=2,
                        depth=2,
                        bayesian=False,
                    ).to(config.device)
                    self.ensembles.append(model)

            def fmodel(params, buffers, x):
                return functional_call(self.ensembles[0], (params, buffers), x)

            self.predict_ensemble = vmap(fmodel, in_dims=(0, 0, None))

            self.optimizer = optim.Adam(
                self.params.values(), lr=self.learning_rate, weight_decay=1e-5, fused=False
            )
            self.scaler = torch.amp.GradScaler(device=config.device)

            print(f'Successfully loaded model weights of {self.ensemble_size} ensembles.')
            
            self.get_available_galaxies()
            
            df_scores = pd.read_csv(os.path.join(round_path, 'probs.csv'), low_memory=False)
            names = df_scores['name'].values.tolist()
            probs = df_scores['prob'].values.tolist()
            uncertainties = df_scores['uncertainty'].values.tolist()
            
            self.scores = {str(name): float(prob) for name, prob in zip(names, probs)}
            self.uncertainties = {str(name): float(uncertainty) for name, uncertainty in zip(names, uncertainties)}
            
            self.sl_scores = [self.scores[str(name)] for name in self.selected_sl_names]
            self.non_sl_scores = [self.scores[str(name)] for name in self.selected_non_sl_names]
            
            print(f'Loaded history from {round_path}.')
            print('Current round: ', self.current_round)

            self.current_round += 1

    def reset_round(self):
        
        previous_round = self.current_round - 1
        
        if previous_round < 0:
            
            # at initial round
            self.selected_sl_names = self.cowls_sl_names.copy()
            self.selected_non_sl_names = []
            
            if os.path.exists(self.testing_data_path):
                with open(self.testing_data_path, 'r') as f:
                    testing_data = json.load(f)
                
                self.testing_sl_names = testing_data['testing_sl_names']
                self.testing_non_sl_names = testing_data['testing_non_sl_names']
            else:
                self.testing_sl_names = self.injected_names.copy()
                self.testing_non_sl_names = []

            self.total_submissions = 0
            self.last_sl_count = len(self.selected_sl_names)
            
        
        else:
            
            round_path = os.path.join(self.results_path, f'round_{previous_round}')
            records_path = os.path.join(round_path, 'records.json')

            with open(records_path, 'r') as f:
                records = json.load(f)
            
            self.selected_sl_names = records['sl_names']
            self.selected_non_sl_names = records['non_sl_names']
            self.testing_sl_names = records['testing_sl_names']
            self.testing_non_sl_names = records['testing_non_sl_names']
            self.injected_recovery = records['injected_recovery']
            self.total_submissions = records['total_submissions']
            
            self.last_sl_count = records['sl_count']
            
        
    def reset_model(self):
        
        self.model_resetted = True
        print('Resetting ensembles...')
        self.initialize_ensembles()
