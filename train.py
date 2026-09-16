"""Training and evaluation entry point for TVI-Net."""

import argparse
import sys

import torch

from datasets.dataloader import MatDataset
from models.TVINet import TVINet
from utils.util import calc_res, get_logger, output_args, setup_seed


def _provided_options(argv):
    provided = set()
    for item in argv:
        if item.startswith("--"):
            provided.add(item.split("=")[0])
    return provided


def _set_dataset_default(args, provided, name, value):
    option = "--" + name.replace("_", "-")
    if option not in provided:
        setattr(args, name, value)


def _sync_formula_aliases(args):
    if getattr(args, "vae_loss_weight", None) is not None:
        args.alpha = args.vae_loss_weight
    if getattr(args, "trust_loss_weight", None) is not None:
        args.lambda_t = args.trust_loss_weight

    args.vae_loss_weight = args.alpha
    args.trust_loss_weight = args.lambda_t
    return args


def get_args(dataset_name="corel5k", argv=None):
    parsed_argv = sys.argv[1:] if argv is None else list(argv)
    provided = _provided_options(parsed_argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="corel5k")
    parser.add_argument("--training-ratio", type=float, default=0.7)
    parser.add_argument("--missing-rate-data", type=float, default=0.5)
    parser.add_argument("--missing-rate-label", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--fold-idx", type=int, default=0)
    parser.add_argument("--ins-temperature", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=5e-2, help="Weight for L_VAE in the paper formula")
    parser.add_argument("--beta", type=float, default=0.9, help="Weight for L_rec in the paper formula")
    parser.add_argument("--gamma", type=float, default=5e-2, help="Weight for L_con in the paper formula")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--mask-rate", type=float, default=0.3)
    parser.add_argument("--mid-dim", type=int, default=1024)
    parser.add_argument("--layer-num", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--run-times", type=int, default=10)
    parser.add_argument("--early-stop", type=bool, default=True)
    parser.add_argument("--use-trustworthy", type=bool, default=True, help="Enable the TVI-Net auxiliary trustworthy multi-label classification branch")
    parser.add_argument(
        "--trustworthy-fusion",
        type=str,
        default="trust",
        choices=["attention", "gated", "trust"],
        help="Fusion type for trustworthy module",
    )
    parser.add_argument("--trust-temperature", type=float, default=0.1, help="Temperature for trust score softmax")
    parser.add_argument("--uncertainty-weight", type=float, default=0.1, help="Weight for uncertainty penalty")
    parser.add_argument("--trust-consistency-weight", type=float, default=0.05, help="Weight for trust consistency loss")
    parser.add_argument("--lambda-t", dest="lambda_t", type=float, default=1.0, help="Weight for L_trust in the paper formula")
    parser.add_argument("--trust-loss-weight", type=float, default=None, help="Deprecated alias for --lambda-t")
    parser.add_argument("--uncertainty-loss-weight", type=float, default=0.1, help="Weight for uncertainty loss")

    parser.add_argument("--use-vae-imputation", type=bool, default=True, help="Enable VAE-based View Imputation (framework a)")
    parser.add_argument("--vae-hidden-dim", type=int, default=512, help="Hidden dimension for VAE")
    parser.add_argument("--vae-latent-dim", type=int, default=256, help="Latent dimension for VAE")
    parser.add_argument("--vae-loss-weight", type=float, default=None, help="Deprecated alias for --alpha")
    parser.add_argument("--vae-recon-weight", type=float, default=1.0, help="Reconstruction weight for VAE")
    parser.add_argument("--vae-kl-weight", type=float, default=0.01, help="KL divergence weight for VAE")
    parser.add_argument("--use-artificial-mask", type=bool, default=True, help="Enable artificial mask for masked-view reconstruction training")
    parser.add_argument("--artificial-mask-ratio", type=float, default=0.2, help="Ratio of artificial masking for observed views")
    parser.add_argument("--lambda-masked-recon", type=float, default=0.3, help="Weight for masked reconstruction loss in main decoder")

    args = parser.parse_args(argv)
    args.dataset_name = dataset_name

    if dataset_name == "corel5k":
        args.hidden = 768
        _set_dataset_default(args, provided, "alpha", 3e-2)
        _set_dataset_default(args, provided, "beta", 0.7)
        _set_dataset_default(args, provided, "gamma", 3e-2)
        args.mask_rate = 0.5
    elif dataset_name == "pascal07":
        args.layer_num = 2
        _set_dataset_default(args, provided, "gamma", 1e-1)
        args.ins_temperature = 0.1
    elif dataset_name == "espgame":
        _set_dataset_default(args, provided, "gamma", 3e-2)
    elif dataset_name == "iaprtc12":
        _set_dataset_default(args, provided, "gamma", 1e-2)
    elif dataset_name == "mirflickr":
        _set_dataset_default(args, provided, "gamma", 1e-2)

    return _sync_formula_aliases(args)


def run_experiment(args, log_dir="results/logs/", record_dir="results/records/"):
    final_res = []
    for idx in range(args.run_times):
        setup_seed(args.seed)
        args.fold_idx = idx

        print("Loading {} dataset.....".format(args.dataset_name))
        train_dataset = MatDataset(args, mode="train")
        test_dataset = MatDataset(args, mode="test")
        val_dataset = MatDataset(args, mode="val")
        print("Loading {} dataset over!!!".format(args.dataset_name))

        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.view_num = train_dataset.view_num
        args.class_num = train_dataset.class_num
        args.total_samples = train_dataset.total_samples
        args.input_dims = []
        for i in range(args.view_num):
            args.input_dims.append(train_dataset.data[i].shape[1])

        logger = get_logger(args, main_dir=log_dir)
        try:
            logger.info(f"--------------------TVI-Net training {args.fold_idx + 1} time--------------------")

            output_args(logger, args)

            model = TVINet(args, logger).to(args.device)
            res = model.run_train(train_dataset, val_dataset, test_dataset)

            final_res.append(res)
        finally:
            logger.handlers.clear()

    logger = get_logger(args, main_dir=record_dir)
    try:
        logger.info("--------------------TVI-Net training over--------------------")

        output_args(logger, args)
        metrics = calc_res(logger, final_res)
    finally:
        logger.handlers.clear()
    return {
        "metrics": metrics,
        "final_res": final_res,
    }


if __name__ == "__main__":
    for dataset_name in ["corel5k"]:
        experiment_args = get_args(dataset_name)
        run_experiment(experiment_args)
