"""
Created on 14:58 at 24/11/2021
@author: bo
"""
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def give_args():
    parser = argparse.ArgumentParser(description='ViT')
    parser.add_argument('--dataset', type=str, default="CIFAR10")
    parser.add_argument("--model_type", type=str, default="eetti16")
    parser.add_argument('--batch_size', type=int, default=120)
    parser.add_argument('--grad_norm_clip', type=float, default=1.0)
    parser.add_argument('--prefetch', type=int, default=2)
    parser.add_argument('--enc_lr', type=float, default=0.03)
    parser.add_argument('--decay_step', type=str, default="cosine")
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--num_gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--add_positional_encoding", type=str2bool, default=True)
    parser.add_argument("--patch_height", type=int, default=2)
    parser.add_argument("--patch_width", type=int, default=2)
    parser.add_argument("--normalization", type=str, default="none")

    parser.add_argument("--strategy", type=str, default="dp")
    parser.add_argument("--quantification", type=str2bool, default=False)
    parser.add_argument("--target_h", type=int, default=54)
    parser.add_argument("--target_w", type=int, default=54)
    parser.add_argument("--top_selection_method", type=str, default="sers_maps")
    parser.add_argument("--percentage", type=float, default=0.00)
    parser.add_argument("--concentration_float", type=float, default=1e-6)
    parser.add_argument("--detection", type=str2bool, default=True)

    parser.add_argument("--avg_spectra", type=str2bool, default=True)
    parser.add_argument("--seed_use", type=int, default=42)
    parser.add_argument("--augment_seed_use", type=int, default=24907)
    parser.add_argument("--leave_index", type=int, default=0)
    parser.add_argument("--leave_method", type=str, default="leave_one_chip")
    parser.add_argument("--loc", type=str, default="home")
    parser.add_argument("--quantification_loss", type=str, default="mse")
    parser.add_argument("--data_dir", type=str, default="../rs_dataset/")

    parser.add_argument("--gpu_index", type=int, default=10)
    parser.add_argument("--lr_schedule", type=str, default="cosine")
    parser.add_argument("--model_init", type=str, default="xavier")
    return parser, parser.parse_args()


def get_model_args():
    parser, args = give_args()
    give_eetti16_args(parser)
    return parser.parse_args()


def give_eetti16_args(parser):
    parser.add_argument('--input_feature', type=int, default=192)
    parser.add_argument('--mlp_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--attention_dropout_rate', type=float, default=0.0)
    parser.add_argument('--drpout_rate', type=float, default=0.0)
    parser.add_argument('--pool', type=str, default='cls')
    parser.add_argument('--representation_size', type=int, default=0)


class const:
    patch_size=2
    input_feature=192
    mlp_dim=512
    num_heads=3 
    num_layers=2
    pool="cls"
    representation_size=0
    leave_method="leave_one_chip"
    leave_index=1
    lr=0.2 
    target_shape=[44, 44]
    dataset="DNP"
    detection=True
    quantification=False 
    quantification_loss="none"
    concentration_float=0.0
    

def get_config_test_vit(dataset, detection, quantification, leave_index=30):
    quantification_loss = "mse" if quantification == True else "none"
    if "TOMAS" in dataset:
        target_shape = [56, 56]
        lr = 0.08 if quantification == True else 0.008 
    elif "DNP" in dataset:
        target_shape = [44, 44]
        lr = 0.006 if quantification == True else 0.005
    elif "PA" in dataset:
        target_shape = [40, 40]
        lr = 0.0006
    args = const 
    args.target_shape = target_shape 
    args.lr = lr 
    args.dataset = dataset 
    args.quantification = quantification
    args.detection = detection 
    args.quantification_loss = quantification_loss 
    args.leave_index = leave_index 
    args.concentration_float = 1e-6 if dataset != "PA" else 1e-5
    return args





