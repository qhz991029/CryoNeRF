import os

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pytorch_lightning as pl
import rich
import timm
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import umap
from einops import rearrange, reduce, repeat

from ..utils import positional_encoding
from .deformation import DeformationDecoder, DeformationEncoder
from .nerf import NeuralRadianceField


class CryoNeRF(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        self.size = args.size
        self.batch_size = args.batch_size
        self.ray_num = args.ray_num
        self.nerf_enc_dim = args.nerf_enc_dim
        self.nerf_hid_dim = args.nerf_hid_dim
        self.nerf_hid_layer_num = args.nerf_hid_layer_num
        self.dfom_enc_dim = args.dfom_enc_dim
        self.dfom_hid_dim = args.dfom_hid_dim
        self.dfom_hid_layer_num = args.dfom_hid_layer_num
        self.dfom_encoder = args.dfom_encoder
        self.dfom_latent_dim = args.dfom_latent_dim
        self.save_dir = args.save_dir
        self.log_step = args.log_step
        self.print_step = args.print_step
        self.enable_dfom = args.enable_dfom
        self.training = False
        self.checkpointing = args.checkpointing

        self.nerf = NeuralRadianceField(self.nerf_enc_dim, self.nerf_hid_dim, self.nerf_hid_layer_num, checkpointing=self.checkpointing)
        if self.enable_dfom:
            self.deformation_encoder = DeformationEncoder(self.dfom_encoder, self.dfom_latent_dim)
            self.deformation_decoder = DeformationDecoder(self.dfom_enc_dim, self.dfom_latent_dim // 2, self.dfom_hid_dim,
                                                          self.dfom_hid_layer_num, checkpointing=self.checkpointing)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.nerf.parameters())

        return {"optimizer": optimizer}
    
    def render_density(self, coords_query, latent_variable=None):
        encoded_xyz = positional_encoding(coords_query, enc_dim=self.dfom_enc_dim)
        # encoded_xyz = positional_encoding_geom(sampled_coords_xyz, size, L)

        if latent_variable is not None:
            delta_xyz = self.deformation_decoder(encoded_xyz, latent_variable)
        else:
            delta_xyz = 0

        deformed_xyz = coords_query + delta_xyz
        encoded_deformed_xyz = positional_encoding(deformed_xyz, enc_dim=self.nerf_enc_dim)
        pred_density = self.nerf(encoded_deformed_xyz)
        
        return pred_density
    
    def reparameterize(self, latent_vecotr, training=True):
        if not training:
            return latent_vecotr[:, :self.dfom_latent_dim // 2]
        mu = latent_vecotr[:, :self.dfom_latent_dim // 2]
        log_var = latent_vecotr[:, self.dfom_latent_dim // 2:]
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        latent_variable = mu + eps * std
        
        return latent_variable, mu, log_var
    
    def on_train_epoch_start(self) -> None:
        self.ray_idx_all = repeat(torch.arange(self.size**2), "HW -> B HW D Dim3",
                                  B=self.batch_size, D=self.size, Dim3=3).long().cuda()

        x = np.linspace(-0.5, 0.5, self.size, endpoint=False)
        y = np.linspace(-0.5, 0.5, self.size, endpoint=False)
        z = np.linspace(-0.5, 0.5, self.size, endpoint=False)
        self.volume_grid = torch.from_numpy(np.stack([coord.flatten()
                                            for coord in np.meshgrid(x, y, z, indexing='xy')], axis=1)).float().cuda()
        
        self.training = True

    def training_step(self, batch, batch_idx):
        R = batch["rotations"]
        t = batch["translations"]

        volume_grid_query = repeat(self.volume_grid, "HWD Dim3 -> B HWD Dim3", B=R.shape[0]).bmm(R) + t.unsqueeze(1).bmm(R)
        volume_grid_query = volume_grid_query.reshape(R.shape[0], self.size**2, self.size, 3)
        
        pred_density = []
        
        if self.enable_dfom:
            latent_vector = self.deformation_encoder(batch["images"].unsqueeze(1))
            latent_variable, mu, log_var = self.reparameterize(latent_vector)
        else:
            latent_variable = None

        for ray_idx in torch.split(self.ray_idx_all, self.ray_num, dim=1):
            sampled_coords_xyz = torch.gather(volume_grid_query, 1, ray_idx)
            pred_density_block = self.render_density(sampled_coords_xyz, latent_variable)
            pred_density.append(pred_density_block.squeeze(-1))

        pred_density = rearrange(torch.cat(pred_density, dim=1), "B (H W) D -> B H W D", H=self.size, W=self.size)
        pred_image = pred_density.mean(-1)
        corrupted_pred_image = torch.fft.fftshift(
            torch.fft.irfft2(
                torch.fft.rfft2(torch.fft.ifftshift(pred_image)) * torch.fft.fftshift(batch["ctfs"])[..., :self.size // 2 + 1]
            )
        )

        # loss = F.mse_loss(pred_image, sample["images"])
        loss_recon = F.mse_loss(corrupted_pred_image, batch["images"])
        if self.enable_dfom:
            loss_kldiv = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        else:
            loss_kldiv = torch.tensor(0)
        loss = loss_recon + loss_kldiv
        
        if batch_idx % self.print_step == 0 and self.trainer.global_rank == 0:
            rich.print(f"Current step: {batch_idx:06d}, loss: {loss:.6f}, loss_recon: {loss_recon:.6f}, loss_kldiv: {loss_kldiv:.6f}")

        if batch_idx % self.log_step == 0 and self.trainer.global_rank == 0:
            log_dir = f"{self.save_dir}/vis/{batch_idx:06d}"
            os.makedirs(log_dir, exist_ok=True)
            plt.imsave(f"{log_dir}/pr.png", pred_image[0].numpy(force=True), cmap="gray")
            plt.imsave(f"{log_dir}/cr.png", corrupted_pred_image[0].numpy(force=True), cmap="gray")
            plt.imsave(f"{log_dir}/gt.png", batch["images"][0].numpy(force=True), cmap="gray")
            plt.imsave(f"{log_dir}/x.png", pred_density[0, self.size // 2].numpy(force=True).transpose(), cmap="gray")
            plt.imsave(f"{log_dir}/y.png", pred_density[0, :, self.size // 2].numpy(force=True).transpose(), cmap="gray")
            plt.imsave(f"{log_dir}/z.png", pred_density[0, :, :, self.size // 2].numpy(force=True).transpose(), cmap="gray")

            with mrcfile.new(f"{log_dir}/volume.mrc", overwrite=True) as mrc:
                mrc.set_data(pred_density[0].numpy(force=True))
                mrc.set_volume()

        return loss
    
    # @torch.no_grad
    # def on_train_epoch_end(self) -> None:
    #     self.size *= 2
    #     x = np.linspace(-0.5, 0.5, self.size, endpoint=False)
    #     y = np.linspace(-0.5, 0.5, self.size, endpoint=False)
    #     z = np.linspace(-0.5, 0.5, self.size, endpoint=False)
    #     volume_grid_query = torch.from_numpy(
    #         np.stack([coord.flatten() for coord in np.meshgrid(x, y, z, indexing='xy')], axis=1)).float().cuda().unsqueeze(0)
    #     volume_grid_query = volume_grid_query.reshape(1, self.size**2, self.size, 3)
        
    #     pred_density = []
        
    #     # latent_variable = self.deformation_encoder(batch["images"].unsqueeze(1))
    #     ray_idx_all = repeat(torch.arange(self.size**2), "HW -> B HW D Dim3", B=1, D=self.size, Dim3=3).long().cuda()
    #     for ray_idx in torch.split(ray_idx_all, self.ray_num, dim=1):
    #         sampled_coords_xyz = torch.gather(volume_grid_query, 1, ray_idx)

    #         # encoded_xyz = positional_encoding(sampled_coords_xyz, enc_dim=self.dfom_enc_dim)
    #         # encoded_xyz = positional_encoding_geom(sampled_coords_xyz, size, L)
            
    #         # delta_xyz = self.deformation_decoder(encoded_xyz, latent_variable)
            
    #         # deformed_xyz = sampled_coords_xyz + delta_xyz
            
    #         encoded_deformed_xyz = positional_encoding(sampled_coords_xyz, enc_dim=self.nerf_enc_dim)

    #         pred_density_block = self.nerf(encoded_deformed_xyz).detach().cpu()

    #         pred_density.append(pred_density_block.squeeze(-1))

    #     pred_density = rearrange(torch.cat(pred_density, dim=1), "B (H W) D -> B H W D", H=self.size, W=self.size)
    #     with mrcfile.new(f"{self.save_dir}/volume.mrc", overwrite=True) as mrc:
    #             mrc.set_data(pred_density[0].numpy(force=True))
    #             mrc.set_volume()
                
    def on_validation_epoch_start(self):
        self.latent_vectors = []
        self.umap_model = umap.UMAP(n_components=2, random_state=42)
        x = np.linspace(-0.5, 0.5, self.size, endpoint=False)
        y = np.linspace(-0.5, 0.5, self.size, endpoint=False)
        z = np.linspace(-0.5, 0.5, self.size, endpoint=False)
        self.volume_grid_query = torch.from_numpy(
            np.stack([coord.flatten() for coord in np.meshgrid(x, y, z, indexing='xy')], axis=1)).float().cuda().unsqueeze(0)
        self.volume_grid_query = self.volume_grid_query.reshape(1, self.size**2, self.size, 3)
            
    @torch.no_grad
    def validation_step(self, batch, batch_idx):
        if self.enable_dfom:
            self.latent_vectors.append(self.deformation_encoder(batch["images"].unsqueeze(1)))
    
    @torch.no_grad
    def on_validation_epoch_end(self):
        if self.trainer.global_rank == 0:
            if self.enable_dfom:
                self.latent_vectors = torch.cat(self.latent_vectors, dim=0)
                ids = torch.from_numpy(np.random.choice(self.latent_vectors.shape[0], size=10))
                latent_variables, mu, std = self.reparameterize(self.latent_vectors)
                latent_2d = self.umap_model.fit_transform(mu.numpy(force=True))
                # plt.scatter(latent_2d[:, 0], latent_2d[:, 1])
                plt.hexbin(latent_2d[:, 0], latent_2d[:, 1], gridsize=100, cmap='Blues')
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.savefig(f'{self.save_dir}/latent.png', dpi=300)
                np.save(f"{self.save_dir}/latent_variables.npy", self.latent_vectors.numpy(force=True))
            else:
                latent_variables = None
            
            ray_idx_all = repeat(torch.arange(self.size**2), "HW -> B HW D Dim3", B=1, D=self.size, Dim3=3).long().cuda()
            
            if self.enable_dfom:
                for i, latent_variable in enumerate(latent_variables[ids, :]):
                    pred_density = []
                    for ray_idx in torch.split(ray_idx_all, self.ray_num, dim=1):
                        sampled_coords_xyz = torch.gather(self.volume_grid_query, 1, ray_idx)
                        pred_density_block = self.render_density(sampled_coords_xyz, latent_variable.unsqueeze(0)).detach().cpu()
                        pred_density.append(pred_density_block.squeeze(-1))

                    pred_density = rearrange(torch.cat(pred_density, dim=1), "B (H W) D -> B H W D", H=self.size, W=self.size)
                    with mrcfile.new(f"{self.save_dir}/volume_{i}.mrc", overwrite=True) as mrc:
                            mrc.set_data(pred_density[0].numpy(force=True))
                            mrc.set_volume()
            else:
                pred_density = []
                for ray_idx in torch.split(ray_idx_all, self.ray_num, dim=1):
                    sampled_coords_xyz = torch.gather(self.volume_grid_query, 1, ray_idx)
                    pred_density_block = self.render_density(sampled_coords_xyz).detach().cpu()
                    pred_density.append(pred_density_block.squeeze(-1))

                pred_density = rearrange(torch.cat(pred_density, dim=1), "B (H W) D -> B H W D", H=self.size, W=self.size)
                with mrcfile.new(f"{self.save_dir}/volume.mrc", overwrite=True) as mrc:
                        mrc.set_data(pred_density[0].numpy(force=True))
                        mrc.set_volume()
        
