import torch

from experiments.loss_utils import calc_loss
from implicit_diff.implicit_diff import Hypergrad


def auxinash_hyperstep(
    model,
    train_loader,
    val_loader,
    weight_method,
    aux_params_alpha,
    aux_params_p,
    n_meta_loss_accum,
    main_task,
    device,
):
    with torch.no_grad():
        aux_params_alpha.data = torch.tensor(
            weight_method.method.prvs_alpha, device=device
        ).data.float()

    meta_val_loss = 0.0
    for _ in range(n_meta_loss_accum):
        # todo: make sure this is randomized each time.
        #  Also, consider using train here as well next...
        batch = next(iter(val_loader))
        batch = (t.to(device) for t in batch)
        val_data, val_label, val_depth, val_normal = batch
        val_label = val_label.long()
        val_pred = model(val_data)

        loss = [
            calc_loss(val_pred[0], val_label, "semantic"),
            calc_loss(val_pred[1], val_depth, "depth"),
            calc_loss(val_pred[2], val_normal, "normal"),
        ][main_task]
        meta_val_loss = meta_val_loss + loss

    total_meta_train_loss = 0.0
    for _ in range(n_meta_loss_accum):
        batch = next(iter(train_loader))
        batch = (t.to(device) for t in batch)
        train_data, train_label, train_depth, train_normal = batch
        train_label = train_label.long()
        train_pred = model(train_data)

        losses = [
            calc_loss(train_pred[0], train_label, "semantic"),
            calc_loss(train_pred[1], train_depth, "depth"),
            calc_loss(train_pred[2], train_normal, "normal"),
        ]
        loss = sum([l * w for l, w in zip(losses, aux_params_alpha)])
        total_meta_train_loss = total_meta_train_loss + loss

    # hypergrad for alpha
    hg = Hypergrad(learning_rate=1e-4)
    alpha_hypergrads = hg.grad(
        loss_val=meta_val_loss,
        loss_train=total_meta_train_loss,
        aux_params=aux_params_alpha,
        params=list(model.auxi_shared_parameters()),
    )

    alpha_hypergrads = list(-g for g in alpha_hypergrads)

    lambda_0 = torch.diag(aux_params_p / aux_params_alpha**2)
    lambda_1 = torch.diag(1 / aux_params_alpha)
    dalpha_dp = torch.inverse(weight_method.method.gtg + lambda_0) @ lambda_1

    hyper_gard = -alpha_hypergrads[0] @ dalpha_dp
    aux_params_p.grad = hyper_gard.data
    return hyper_gard
