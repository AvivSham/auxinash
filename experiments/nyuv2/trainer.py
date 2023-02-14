import logging
from argparse import ArgumentParser

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.clip_grad import clip_grad_norm_

from experiments.loss_utils import calc_loss
from experiments.nyuv2.calc_delta import delta_fn
from tqdm import trange
import wandb

from experiments.nyuv2.data import NYUv2
from implicit_diff.optim import MetaOptimizer
from methods.auxilearn.hyperstep import auxilearn_hyperstep
from methods.auxinash.hyperstep import auxinash_hyperstep
from methods.weight_methods import WeightMethods
from experiments.utils import (
    set_seed,
    set_logger,
    common_parser,
    get_device,
    extract_weight_method_parameters_from_args,
    str2bool,
)
from experiments.nyuv2.models import SegNet
from experiments.nyuv2.utils import ConfMatrix, depth_error, normal_error

set_logger()


def main(args, device):
    # ----
    # Nets
    # ---
    model = SegNet(),
    model = model.to(device)

    # dataset and dataloaders
    log_str = (
        "Applying data augmentation on NYUv2."
        if args.apply_augmentation
        else "Standard training strategy without data augmentation."
    )
    logging.info(log_str)

    nyuv2_train_val_set = NYUv2(
        root=args.data_path.as_posix(), train=True, augmentation=args.apply_augmentation
    )

    nyuv2_test_set = NYUv2(root=args.data_path.as_posix(), train=False)

    if args.partial_data:
        # split to train and val
        train_idx, val_idx = train_test_split(
            range(len(nyuv2_train_val_set)), test_size=args.val_size
        )

        nyuv2_train_set = Subset(nyuv2_train_val_set, train_idx)
        nyuv2_val_set = Subset(nyuv2_train_val_set, val_idx)
    else:
        nyuv2_train_set = nyuv2_train_val_set
        nyuv2_val_set = nyuv2_train_val_set

    logging.info(f"train size: {len(nyuv2_train_set)}, test set: {len(nyuv2_test_set)}")

    train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_val_set, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set, batch_size=args.batch_size, shuffle=False
    )

    # weight method
    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    weight_method = WeightMethods(
        args.method, n_tasks=3, device=device, **weight_methods_parameters[args.method]
    )

    # optimizer
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(
            [
                dict(params=model.parameters(), lr=args.lr),
                dict(params=weight_method.parameters(), lr=args.method_params_lr),
            ],
            momentum=0.9,
            lr=args.lr,
        )
    else:
        optimizer = torch.optim.Adam(
            [
                dict(params=model.parameters(), lr=args.lr),
                dict(params=weight_method.parameters(), lr=args.method_params_lr),
            ],
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    if args.method == "auxinash":
        aux_params_alpha = torch.tensor(
            weight_method.method.prvs_alpha,
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        aux_params_p = torch.tensor(
            weight_method.method.p,
            device=device,
            requires_grad=True,
            dtype=torch.float32,
        )
        meta_opt = torch.optim.SGD(
            [aux_params_p],
            lr=args.meta_lr,
            momentum=0.9,
        )

    if args.method == "auxilearn":
        auxilearn_opt = torch.optim.SGD(
            weight_method.parameters(),
            lr=args.meta_lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

        meta_opt = MetaOptimizer(
            meta_optimizer=auxilearn_opt, hpo_lr=1e-4, truncate_iter=3, max_grad_norm=25
        )

    epochs = args.n_epochs
    epoch_iter = trange(epochs)
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([epochs, 24], dtype=np.float32)
    custom_step = -1
    conf_mat = ConfMatrix(model.segnet.class_nb)
    for epoch in epoch_iter:
        cost = np.zeros(24, dtype=np.float32)

        for j, batch in enumerate(train_loader):
            custom_step += 1

            model.train()
            optimizer.zero_grad()

            train_data, train_label, train_depth, train_normal = batch
            train_data, train_label = train_data.to(device), train_label.long().to(
                device
            )
            train_depth, train_normal = train_depth.to(device), train_normal.to(device)

            train_pred, features = model(train_data, return_representation=True)

            losses = torch.stack(
                (
                    calc_loss(train_pred[0], train_label, "semantic"),
                    calc_loss(train_pred[1], train_depth, "depth"),
                    calc_loss(train_pred[2], train_normal, "normal"),
                )
            )

            loss, extra_outputs = weight_method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features,
            )

            optimizer.step()

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = losses[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(
                train_pred[2], train_normal
            )
            avg_cost[epoch, :12] += cost[:12] / train_batch

            log = {}
            if args.wandb:
                log.update(
                    {
                        "train/semantic_loss": losses[0].item(),
                        "train/depth_loss": losses[1].item(),
                        "train/normal_loss": losses[2].item(),
                    }
                )

                task_weights = extra_outputs.get("weights", None)
                if task_weights is not None:
                    log.update(
                        {
                            f"train/weight_task_{i}": w.item()
                            for i, w in enumerate(task_weights)
                        }
                    )
                if "sigma_min" in extra_outputs:
                    log.update(
                        {
                            "sigma_min": extra_outputs["sigma_min"],
                            "sigma_max": extra_outputs["sigma_max"],
                            "normalized_sigma_min": extra_outputs[
                                "normalized_sigma_min"
                            ],
                        }
                    )

            epoch_iter.set_description(
                f"[{epoch+1}  {j+1}/{train_batch}] semantic loss: {losses[0].item():.3f}, "
                f"depth loss: {losses[1].item():.3f}, "
                f"normal loss: {losses[2].item():.3f}"
            )

            if ((custom_step + 1) % args.hypergrad_every) == 0:
                if args.method == "auxinash":
                    meta_opt.zero_grad()
                    auxinash_hyperstep(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        weight_method=weight_method,
                        aux_params_alpha=aux_params_alpha,
                        aux_params_p=aux_params_p,
                        n_meta_loss_accum=args.n_meta_loss_accum,
                        main_task=args.main_task,
                        device=device,
                    )
                    clip_grad_norm_(aux_params_p, max_norm=1)
                    meta_opt.step()

                    # update p
                    weight_method.method.set_p(aux_params_p.detach().cpu().numpy())
                    # make sure p is aligned between weight method and opt params
                    with torch.no_grad():
                        aux_params_p.data = torch.from_numpy(weight_method.method.p).to(
                            device
                        )
                if args.method == "auxilearn":
                    meta_opt.zero_grad()
                    auxilearn_hyperstep(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        weight_method=weight_method,
                        meta_optimizer=meta_opt,
                        n_meta_loss_accum=args.n_meta_loss_accum,
                        main_task=args.main_task,
                        device=device,
                    )

            if args.method == "auxinash":
                log.update(
                    {
                        f"preference/weight_task_{i}": w
                        for i, w in enumerate(weight_method.method.p)
                    }
                )

            if args.wandb:
                wandb.log(log)

        # scheduler
        scheduler.step()
        # compute mIoU and acc
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        # evaluating test data
        model.eval()
        conf_mat = ConfMatrix(model.segnet.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth, test_normal = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.long().to(
                    device
                )
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred = model(test_data)
                test_loss = torch.stack(
                    (
                        calc_loss(test_pred[0], test_label, "semantic"),
                        calc_loss(test_pred[1], test_depth, "depth"),
                        calc_loss(test_pred[2], test_normal, "normal"),
                    )
                )

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                cost[18] = test_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(
                    test_pred[2], test_normal
                )
                avg_cost[epoch, 12:] += cost[12:] / test_batch

            # compute mIoU and acc
            avg_cost[epoch, 13:15] = conf_mat.get_metrics()

            # print results
            print(
                f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR "
                f"| NORMAL_LOSS MEAN MED <11.25 <22.5 <30"
            )
            print(
                f"Epoch: {epoch:04d} | TRAIN: {avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} "
                f"| {avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} | {avg_cost[epoch, 6]:.4f} "
                f"{avg_cost[epoch, 7]:.4f} {avg_cost[epoch, 8]:.4f} {avg_cost[epoch, 9]:.4f} {avg_cost[epoch, 10]:.4f} {avg_cost[epoch, 11]:.4f} ||"
                f"TEST: {avg_cost[epoch, 12]:.4f} {avg_cost[epoch, 13]:.4f} {avg_cost[epoch, 14]:.4f} | "
                f"{avg_cost[epoch, 15]:.4f} {avg_cost[epoch, 16]:.4f} {avg_cost[epoch, 17]:.4f} | {avg_cost[epoch, 18]:.4f} "
                f"{avg_cost[epoch, 19]:.4f} {avg_cost[epoch, 20]:.4f} {avg_cost[epoch, 21]:.4f} {avg_cost[epoch, 22]:.4f} {avg_cost[epoch, 23]:.4f}"
            )

            if args.wandb:
                delta_m = delta_fn(
                    avg_cost[epoch, [13, 14, 16, 17, 19, 20, 21, 22, 23]]
                )
                eval_log = {
                    "epoch": epoch,
                    "train/semantic_loss": avg_cost[epoch, 0],
                    "train/mean_iou": avg_cost[epoch, 1],
                    "train/pix_acc": avg_cost[epoch, 2],
                    "train/depth_loss": avg_cost[epoch, 3],
                    "train/abs_err": avg_cost[epoch, 4],
                    "train/rel_err": avg_cost[epoch, 5],
                    "train/normal_loss": avg_cost[epoch, 6],
                    "train/normal_mean": avg_cost[epoch, 7],
                    "train/normal_med": avg_cost[epoch, 8],
                    "train/normal_<11.25": avg_cost[epoch, 9],
                    "train/normal_<22.5": avg_cost[epoch, 10],
                    "train/normal_<30": avg_cost[epoch, 11],
                    # test
                    "test/delta_m": delta_m,
                    "test/semantic_loss": avg_cost[epoch, 12],
                    "test/mean_iou": avg_cost[epoch, 13],
                    "test/pix_acc": avg_cost[epoch, 14],
                    "test/depth_loss": avg_cost[epoch, 15],
                    "test/abs_err": avg_cost[epoch, 16],
                    "test/rel_err": avg_cost[epoch, 17],
                    "test/normal_loss": avg_cost[epoch, 18],
                    "test/normal_mean": avg_cost[epoch, 19],
                    "test/normal_med": avg_cost[epoch, 20],
                    "test/normal_<11.25": avg_cost[epoch, 21],
                    "test/normal_<22.5": avg_cost[epoch, 22],
                    "test/normal_<30": avg_cost[epoch, 23],
                }
                wandb.log(eval_log)


if __name__ == "__main__":
    parser = ArgumentParser("NYUv2", parents=[common_parser])
    parser.set_defaults(
        data_path="/cortex/data/images/NYUv2/nyuv2",
        lr=1e-4,
        n_epochs=200,
        batch_size=2,
        method="asymnash",
    )
    # hypergrad
    parser.add_argument(
        "--meta-lr",
        type=float,
        default=5e-3,
        help="learning rate for subspace optimizer",
    )
    parser.add_argument(
        "--n-meta-loss-accum",
        type=int,
        default=1,
        help="Num. batches to accum for meta grad calculations",
    )
    parser.add_argument(
        "--hypergrad-every", type=int, default=25, help="Hypergrad freq."
    )

    parser.add_argument("--val-size", type=float, default=0.025, help="meta val size")
    parser.add_argument(
        "--partial-data",
        type=str2bool,
        default=False,
        help="remove meta val from training",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="optimizer type",
    )
    parser.add_argument(
        "--apply-augmentation", type=str2bool, default=True, help="data augmentations"
    )
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    # wandb
    if args.wandb:
        name = f"nyuv2_{args.method}_{args.model}_lr_{args.lr}" f"_seed_{args.seed}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=name,
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(args)

    device = get_device(gpus=args.gpu)
    main(args=args, device=device)
