import os
import torch
from torchvision.utils import save_image, make_grid


def sample_images(G_S2T, G_T2S, S_loader, T_loader, path, steps_done):
    """
    save imgs
    args:
        steps_done: training step
    """
    with torch.no_grad():
        S_imgs, _ = next(iter(S_loader))
        T_imgs, _ = next(iter(T_loader))
        G_S2T.eval()
        G_T2S.eval()
        real_S = S_imgs.cuda()
        fake_T = G_S2T(real_S)
        recv_S = G_T2S(fake_T)
        real_T = T_imgs.cuda()
        fake_S = G_T2S(real_T)
        recv_T = G_S2T(fake_S)

        real_S = make_grid(real_S, nrow=1, normalize=True)
        real_T = make_grid(real_T, nrow=1, normalize=True)
        recv_S = make_grid(recv_S, nrow=1, normalize=True)
        fake_T = make_grid(fake_T, nrow=1, normalize=True)
        fake_S = make_grid(fake_S, nrow=1, normalize=True)
        recv_T = make_grid(recv_T, nrow=1, normalize=True)
        image_grid = torch.cat((real_S, fake_T, recv_S, real_T, fake_S, recv_T), 2)

        save_image(image_grid, os.path.join(path, f"{steps_done:06d}.png") , normalize=False)