import argparse
import json
import os
from types import SimpleNamespace

def str_to_bool(value):
    """Convert string to boolean value."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    return bool(value)

# Parse command line arguments
parser = argparse.ArgumentParser()

# paths
parser.add_argument('--data_path', type=str, 
                    default='../updated_datasets/embeddings.npz', 
                    help='Path to the embeddings file')
parser.add_argument('--images_path', type=str, 
                    default='../updated_datasets/JWST_images', 
                    help='Path to the images directory')
parser.add_argument('--dataframe_path', type=str, 
                    default='../updated_datasets/JWST_SL_discovery_catalog_with_embedding_index.csv', 
                    help='Path to the dataframe file')
parser.add_argument('--rgb_path', type=str, 
                    default='../updated_datasets/JWST_rgbs',
                    help='Path to the RGB images directory')
parser.add_argument('--results_path', type=str, default='results',
                    help='Path to the results directory')
parser.add_argument('--testing_data_path', type=str, 
                    default='hold_out_samples/testing_data.json',
                    help='Path to the testing data file')

# data parameters
parser.add_argument('--latents_scaled', type=str_to_bool,
                    default=False,
                    help='Whether to scale the latents')
parser.add_argument('--score_limit', type=str,
                    default=None, 
                    help='COWLS score limit')
parser.add_argument('--num_injections', type=int,
                    default=0,
                    help='Number of injections')
parser.add_argument('--field', type=str, default='web', choices=['spring', 'web'])
parser.add_argument('--use_focal_loss', type=str_to_bool,
                    default=False,
                    help='Whether to use focal loss')
parser.add_argument('--focal_loss_alpha', type=float,
                    default=0.25,
                    help='Alpha for focal loss')
parser.add_argument('--focal_loss_gamma', type=float,
                    default=2.0,
                    help='Gamma for focal loss')

# model parametersc
parser.add_argument('--embedding_size', type=int,
                    default=256,
                    help='Embedding size')
parser.add_argument('--maximum_ensemble_size', type=int,
                    default=5,
                    help='Maximum ensemble size')
parser.add_argument('--patience', type=int, 
                    default=40,
                    help='Patience')
parser.add_argument('--warmup_epochs', type=int,
                    default=300,
                    help='Warmup epochs')
parser.add_argument('--batch_size', type=int,
                    default=2048,
                    help='Batch size')
parser.add_argument('--learning_rate', type=float,
                    default=1e-3, help='Learning rate')
parser.add_argument('--norm_method', type=str,
                    default='layer',
                    help='Normalization method')
parser.add_argument('--monitor_metric', type=str, 
                    choices=['train_loss', 'test_loss', 'separation'], 
                    default='test_loss', 
                    help='Metric for LR schedule & checkpointing: separation = median rank(non-SL) '
                         '- median rank(SL) within the hold-out test pool (maximize).')
parser.add_argument('--checkpoint_min_epoch', type=int, default=0,
                    help='Minimum epoch before accepting a new best checkpoint (0 = disabled). '
                         'Use this when test-loss early minima cause under-training.')

# other parameters
parser.add_argument('--supplement_ratio', type=float, 
                    default=0.2,
                    help='Supplement ratio')
parser.add_argument('--supplement_method', type=str,
                    default='uncertainty',
                    help='Supplement method')
parser.add_argument('--num_submission_train', type=int,
                    default=100,
                    help='Number of submission train')
parser.add_argument('--random_seed', type=int, default=42, 
                    help='Random seed')
parser.add_argument('--fix_ensembles', type=str_to_bool,
                    default=True,
                    help='Whether to fix the ensembles')
parser.add_argument('--cold_start', type=str_to_bool,
                    default=False,
                    help='Whether to use cold start')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device')

# resume from checkpoint
parser.add_argument('--host', type=str, default='0.0.0.0',
                    help='Host to bind (e.g. 0.0.0.0 for all interfaces, or 10.0.10.80 for a specific IP)')
parser.add_argument('--port', type=int, default=6543,
                    help='Port number')
parser.add_argument('--checkpoint_round', type=int, default=None,
                    help='Checkpoint round')

args = parser.parse_args()

config = args

os.makedirs(config.results_path, exist_ok=True)

# if use checkpoint, load existing configs
if config.checkpoint_round is not None:
    
    print(f'Loading configs')
    with open(f'{config.results_path}/config.json', 'r') as f:
        saved_config = json.load(f)
    
    # ensure host exists (for configs saved before --host was added)
    if 'host' not in saved_config:
        saved_config['host'] = getattr(config, 'host', '0.0.0.0')
    # modify configs by input args
    for key, value in vars(config).items():
        if key in saved_config and saved_config[key] != value:
            print(f'Modify {key}: {saved_config[key]} -> {value}')
            saved_config[key] = value
    
    # final config from saved configs and input args
    config = SimpleNamespace(**saved_config)
    

# save final configs
with open(f'{config.results_path}/config.json', 'w') as f:
    json.dump(vars(config), f, indent=4)
