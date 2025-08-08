import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils
from dataset import get_mnist_dataloader
from model import FMUNet
from flow_matching import FlowMatching


log_savepath = "results/log.txt"
img_savepath = "results/images/"
checkpoint_savepath = "results/checkpoints"
utils.create_parent_directory(log_savepath)


# function to save training log information
def write_log_message(
        global_step: Optional[int] = None,
        lr: Optional[float] = None,
        avg_loss: Optional[float] = None,
        log_message: Optional[str] = None
):
    if log_message is None:
        current_time = time.asctime(time.localtime(time.time()))
        batch_info = f"Global Step: {global_step}"
        loss_info = f"Loss: {avg_loss:.6f}"
        lr_info = f"Learning Rate: {lr:.6f}"

        log_message = f"{current_time}\n{batch_info}, {loss_info}, {lr_info}\n\n"

    with open(log_savepath, mode='a') as file:
        file.write(log_message)


# function to test generation effects
def test_image(
        model: nn.Module,
        flow_matching: FlowMatching,
        filename: str,
        device: torch.device,
):
    savepath = img_savepath + filename + '.png'
    utils.create_parent_directory(savepath)
    labels = torch.arange(0, 11, device=device)
    images = flow_matching.sample(
        model = model,
        num_timesteps = 100,
        num_imgs = 11,
        labels = labels,
        guidance_scale = 5.0,
        device = device
    )
    utils.visualize(
        tensor = images,
        save_path = savepath,
        title = filename,
        max_images = 11
    )


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


# train loop
def train():
    """
    Basic setup for training
    """
    total_steps = 5000
    batch_size = 512

    # set up accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = 1,
        mixed_precision = 'fp16'
    )
    msg_accel = f"Device: {accelerator.device}, Process: {accelerator.process_index}\n"
    msg_accel += "Grad Accumulation Steps: 1, Grad Clip: 1.0, Mixed Precision: 'fp16'"

    # set up model, optimizer, scheduler and dataloader
    model = FMUNet(
        channels = [1, 32, 64, 128],
        num_residual_layers = 2,
        t_embed_dim = 64,
        y_embed_dim = 16,
        num_classes = 10
    )
    msg_model = f"Model Size: {utils.model_size(model):.6f} MiB"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = 1e-4,
        weight_decay = 0.0
    )
    msg_optim = f"Optimizer: AdamW, lr = 1e-4, weight_decay = 0.0"
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    msg_sched = f"Scheduler: CosineAnneal"
    dataloader, _ = get_mnist_dataloader(
        batch_size = batch_size,
        num_workers = 8,
        train = True,
        test = False
    )
    model, optimizer, scheduler, dataloader = accelerator.prepare(
        model, optimizer, scheduler, dataloader
    )
    dataloader = cycle(dataloader)

    # set up the flow matching part
    flow_matching = FlowMatching(
        num_classes = 10,
        cfg_ratio = 0.1,
        img_shape = (1, 32, 32)
    )

    setup_message = f"{msg_accel}\n{msg_model}\n{msg_optim}\n{msg_sched}\n\n"
    write_log_message(log_message=setup_message)
    print(setup_message)


    """
    Train begin here
    """
    losses = 0.0
    log_step = 100
    test_step = 1000

    with tqdm(
            range(1, total_steps + 1),
            disable = not accelerator.is_main_process,
            dynamic_ncols = True
    ) as pbar:
        pbar.set_description("Training")
        model.train()

        for global_step in pbar:
            with accelerator.accumulate(model):
                images, labels = next(dataloader)
                loss = flow_matching.loss(model, images, labels)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            losses += loss.item()
            pbar.set_postfix(loss=loss.item())

            if accelerator.is_main_process:
                if global_step % log_step == 0:
                    write_log_message(
                        global_step = global_step,
                        lr = optimizer.param_groups[0]['lr'],
                        avg_loss = losses / log_step
                    )
                    losses = 0.0

                if global_step % test_step == 0:
                    model.eval()
                    test_image(
                        model = model,
                        flow_matching = flow_matching,
                        filename = f"test_at_stage_{global_step}_steps",
                        device = accelerator.device
                    )
                    model.train()

    if accelerator.is_main_process:
        accelerator.save_state(checkpoint_savepath)
        accelerator.print(f"Saved final checkpoint to {checkpoint_savepath}")
    return None


if __name__ == '__main__':
    train()
