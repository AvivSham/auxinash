import torch

from experiments.loss_utils import calc_loss


def auxilearn_hyperstep(
    model,
    train_loader,
    val_loader,
    weight_method,
    meta_optimizer,
    n_meta_loss_accum,
    main_task,
    device,
):
    meta_val_loss = 0.0
    for _ in range(n_meta_loss_accum):
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

        losses = torch.stack(losses)
        loss, _ = weight_method.get_weighted_loss(losses)
        total_meta_train_loss = total_meta_train_loss + loss

    hyper_gard = meta_optimizer.step(
        val_loss=meta_val_loss,
        train_loss=total_meta_train_loss,
        aux_params=list(weight_method.parameters()),
        parameters=list(model.shared_parameters()),
        return_grads=True,
    )
    return hyper_gard
