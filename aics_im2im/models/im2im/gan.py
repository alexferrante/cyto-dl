from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from monai.data.meta_tensor import MetaTensor
from monai.inferers import sliding_window_inference

from aics_im2im.models.base_model import BaseModel


class GAN(BaseModel):
    """Basic GAN model."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        task_heads: Dict[str, nn.Module],
        discriminator: nn.Module,
        x_key: str,
        save_dir="./",
        save_images_every_n_epochs=1,
        automatic_optimization: bool = False,
        inference_args: Dict = {},
        **base_kwargs,
    ):
        """
        Parameters
        ----------
        backbone: nn.Module
            backbone network, parameters are shared between task heads
        task_heads: Dict
            task-specific heads
        discriminator
            discriminator network
        x_key: str
            key of input image in batch
        save_dir="./"
            directory to save images during training and validation
        save_images_every_n_epochs=1
            Frequency to save out images during training
        inference_args: Dict = {}
            Arguments passed to monai's [sliding window inferer](https://docs.monai.io/en/stable/inferers.html#sliding-window-inference)
        **base_kwargs:
            Additional arguments passed to BaseModel
        """

        super().__init__(**base_kwargs)
        self.automatic_optimization = False
        for stage in ("train", "val", "test", "predict"):
            (Path(save_dir) / f"{stage}_images").mkdir(exist_ok=True, parents=True)
        self.backbone = backbone
        self.discriminator = discriminator
        self.task_heads = torch.nn.ModuleDict(task_heads)
        assert len(self.task_heads.keys()) == 1, "Only single-head GANs are supported currently."
        for k, head in self.task_heads.items():
            head.update_params({"head_name": k, "x_key": x_key, "save_dir": save_dir})

    def configure_optimizers(self):
        opts = []
        scheds = []
        for key in ("generator", "discriminator"):
            if key in self.optimizer.keys():
                if key == "generator":
                    opt = self.optimizer[key](
                        list(self.backbone.parameters()) + list(self.task_heads.parameters())
                    )
                elif key == "discriminator":
                    opt = self.optimizer[key](self.discriminator.parameters())
                scheduler = self.lr_scheduler[key](optimizer=opt)
                opts.append(opt)
                scheds.append(scheduler)
        return (opts, scheds)

    def _train_forward(self, batch, stage, save_image, run_heads):
        """during training we are only dealing with patches,so we can calculate per-patch loss,
        metrics, postprocessing etc."""
        z = self.backbone(batch[self.hparams.x_key])
        return {
            task: self.task_heads[task].run_head(
                z, batch, stage, save_image, self.global_step, self.discriminator
            )
            for task in run_heads
        }

    def forward(self, x, run_heads):
        z = self.backbone(x)
        return {task: self.task_heads[task](z) for task in run_heads}

    def _inference_forward(self, batch, stage, save_image, run_heads):
        """during inference, we need to calculate per-fov loss/metrics/postprocessing.

        To avoid storing and passing to each head the intermediate results of the backbone, we need
        to run backbone + taskheads patch by patch, then do saving/postprocessing/etc on the entire
        fov.
        """
        with torch.no_grad():
            raw_pred_images = sliding_window_inference(
                inputs=batch[self.hparams.x_key],
                predictor=self.forward,
                run_heads=run_heads,
                **self.hparams.inference_args,
            )
        return {
            head_name: head.run_head(
                None,
                batch,
                stage,
                save_image,
                self.global_step,
                discriminator=self.discriminator if stage == "test" else None,
                run_forward=False,
                y_hat=raw_pred_images[head_name],
            )
            for head_name, head in self.task_heads.items()
        }

    def run_forward(self, batch, stage, save_image, run_heads):
        if stage in ("train", "val"):
            return self._train_forward(batch, stage, save_image, run_heads)
        return self._inference_forward(batch, stage, save_image, run_heads)

    def should_save_image(self, batch_idx, stage):
        return stage in ("test", "predict") or (
            batch_idx == 0  # noqa: FURB124
            and (self.current_epoch + 1) % self.hparams.save_images_every_n_epochs == 0
        )

    def _sum_losses(self, losses):
        summ = 0
        for k, v in losses.items():
            summ += v
        losses["loss"] = summ
        return losses

    def _get_run_heads(self, batch, stage):
        if stage not in ("test", "predict"):
            run_heads = [key for key in self.task_heads.keys() if key in batch]
        else:
            run_heads = self.task_heads.keys()
        return run_heads

    def _extract_loss(self, outs, loss_type):
        loss = {
            f"{head_name}_{loss_type}": head_result[loss_type]
            for head_name, head_result in outs.items()
        }
        return self._sum_losses(loss)

    def _step(self, stage, batch, batch_idx, logger):
        # convert monai metatensors to tensors
        for k, v in batch.items():
            if isinstance(v, MetaTensor):
                batch[k] = v.as_tensor()

        run_heads = self._get_run_heads(batch, stage)
        outs = self.run_forward(batch, stage, self.should_save_image(batch_idx, stage), run_heads)
        if stage == "predict":
            return
        loss_D = self._extract_loss(outs, "loss_D")
        loss_G = self._extract_loss(outs, "loss_G")

        self.log_dict(
            {f"{stage}_{k}": v for k, v in loss_D.items()},
            logger=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )
        self.log_dict(
            {f"{stage}_{k}": v for k, v in loss_G.items()},
            logger=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )
        if stage == "train":
            g_opt, d_opt = self.optimizers()

            g_opt.zero_grad()
            self.manual_backward(loss_G["loss"])
            g_opt.step()

            d_opt.zero_grad()
            self.manual_backward(loss_D["loss"])
            d_opt.step()
