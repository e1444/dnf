# Attempt 1
Separate space
```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.01 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=4 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.freeze_steps.start=4 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512
```

Reshape latent space to match latent distributions 
```sh
sbatch scripts/train_fir.sh \
    training.epochs=10 \
    training.lambda_=0.99 \
    training.beta_final=1.0 \
    training.lr=0 \
    training.lr_mu=0.001 \
    training.lr_v=0.001 \
    training.lr_U=0.001 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-20/20-27-24/checkpoint_epoch_20.pth
```

Reshape latent space to match latent distributions again
```sh
sbatch scripts/train_fir.sh \
    training.epochs=10 \
    training.lambda_=0.99 \
    training.beta_final=1.0 \
    training.lr=0 \
    training.lr_mu=0.001 \
    training.lr_v=0.001 \
    training.lr_U=0.001 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-20/21-03-26/checkpoint_epoch_30.pth
```

Reshape latent space to match latent distributions
```sh
sbatch scripts/train_fir.sh \
    training.epochs=20 \
    training.lambda_=0.99 \
    training.beta_final=1.0 \
    training.r_logdet=0.001 \
    training.aux_total=0 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=4 \
    model.hidden_channels=512 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-20/21-47-28/checkpoint_epoch_40.pth
```

# Attempt 2
```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.01 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=4 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.latent_v=-7.8535 \
    training.latent_noise=0.0001 \
    training.freeze_steps.start=4 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512
```

Reshape latent distributions to match latent space
```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.99 \
    training.beta_final=1.0 \
    training.lr=0 \
    training.lr_mu=0.001 \
    training.lr_v=0.001 \
    training.lr_U=0.001 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-20/22-14-33/checkpoint_epoch_20.pth
```

Reshape latent space to match latent distributions
```sh
sbatch scripts/train_fir.sh \
    training.epochs=20 \
    training.lambda_=0.99 \
    training.beta_final=1.0 \
    training.r_logdet=0.001 \
    training.aux_total=0 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=4 \
    model.hidden_channels=512 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-20/22-46-17/checkpoint_epoch_40.pth
```

Try to fix the separation again haha
```sh
sbatch scripts/train_fir.sh \
    training.epochs=20 \
    training.lambda_=0.5 \
    training.beta_final=1.0 \
    training.r_logdet=0.001 \
    training.aux_total=0 \
    training.freeze_steps.start=4 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-20/23-29-51/checkpoint_epoch_60.pth
```

# Attempt 3
```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.01 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=4 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.latent_v=-8 \
    training.latent_noise=0.0001 \
    training.freeze_steps.start=4 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512
```

Reshape latent distributions to match latent space
```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.99 \
    training.beta_final=1.0 \
    training.lr=0 \
    training.lr_mu=0.001 \
    training.lr_v=0.001 \
    training.lr_U=0.001 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-20/23-32-37/checkpoint_epoch_20.pth
```

# Attempt 4
```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.01 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=4 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.latent_v=-8 \
    training.latent_noise=0.0001 \
    training.freeze_steps.start=4 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512
```

Reshape latent space to match latent distributions
```sh
sbatch scripts/train_fir.sh \
    training.epochs=20 \
    training.lambda_=0.99 \
    training.beta_final=1.0 \
    training.r_logdet=0.001 \
    training.aux_total=0 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=4 \
    model.hidden_channels=512 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-20/23-32-37/checkpoint_epoch_20.pth
```

# Attempt 5
```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.01 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=4 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.latent_v=-8 \
    training.latent_noise=0.0001 \
    training.freeze_steps.start=4 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512
```

Reshape latent space to match latent distributions
```sh
sbatch scripts/train_fir.sh \
    training.epochs=20 \
    training.lambda_=0.99 \
    training.beta_final=1.0 \
    training.lr_mu=0.001 \
    training.lr_v=0.001 \
    training.lr_U=0.001 \
    training.r_logdet=0.001 \
    training.aux_total=0 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=4 \
    model.hidden_channels=512 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-20/23-32-37/checkpoint_epoch_20.pth
```

# Attempt 6
```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.01 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=4 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.latent_separation=23.4409 \
    training.latent_v=-1.1789 \
    training.latent_noise=0.0001 \
    training.freeze_steps.start=4 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512
```

Reshape latent distributions to match latent space
```sh
sbatch scripts/train_fir.sh \
    training.lambda_=1.0 \
    training.beta_final=1.0 \
    training.lr=0 \
    training.lr_mu=0.001 \
    training.lr_v=0.001 \
    training.lr_U=0.001 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-21/02-15-13/checkpoint_epoch_20.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.99 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.lr_mu=0.001 \
    training.lr_v=0.001 \
    training.lr_U=0.001 \
    training.aux_total=4 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.freeze_steps.start=4 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-21/02-43-15/checkpoint_epoch_40.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.99 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.lr=1e-4 \
    training.lr_mu=1e-4 \
    training.lr_v=1e-4 \
    training.lr_U=1e-4 \
    training.aux_total=4 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.freeze_steps.start=4 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-21/10-09-58/checkpoint_epoch_60.pth
```

# Attempt 7
```sh
sbatch scripts/train_fir.sh \
    training.lr=1e-3 \
    training.lr_mu=1e-3 \
    training.lr_U=1e-3 \
    training.lambda_=0.01 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=6 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.freeze_steps.start=6 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.latent_separation=0 \
    training.latent_v=-1.1789 \
    training.latent_noise=0.0001 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3
```

```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.01 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=6 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.freeze_steps.start=6 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-21/14-04-46/checkpoint_epoch_20.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.99 \
    training.beta_final=1.0 \
    training.r_logdet=0 \
    training.aux_total=0 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=6 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-21/14-42-18/checkpoint_epoch_40.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.99 \
    training.beta_final=1.0 \
    training.lr_mu=0.001 \
    training.lr_v=0.001 \
    training.lr_U=0.001 \
    training.r_logdet=0.001 \
    training.aux_total=0 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=4 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-21/14-42-18/checkpoint_epoch_40.pth
```

# Attempt 8
```sh
sbatch scripts/train_fir.sh \
    training.lr=1e-3 \
    training.lr_mu=1e-3 \
    training.lr_U=1e-3 \
    training.lambda_=0.01 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=4 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.freeze_steps.start=4 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.latent_separation=0 \
    training.latent_v=-1.1789 \
    training.latent_noise=0.0001 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3
```

```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.01 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=4 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.freeze_steps.start=4 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-21/14-04-46/checkpoint_epoch_20.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.lambda_=1.00 \
    training.lr=0 \
    training.lr_mu=0.001 \
    training.lr_v=0.001 \
    training.lr_U=0.001 \
    training.beta_final=1.0 \
    training.r_logdet=0 \
    training.aux_total=0 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-21/14-42-18/checkpoint_epoch_40.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.lambda_=1.00 \
    training.lr=0 \
    training.lr_mu=0.001 \
    training.lr_v=0.001 \
    training.lr_U=0.001 \
    training.beta_final=1.0 \
    training.r_logdet=0 \
    training.aux_total=0 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-21/16-41-42/checkpoint_epoch_60.pth
```

# Attempt 9
```sh
sbatch scripts/train_fir.sh \
    training.lr=1e-3 \
    training.lr_mu=1e-3 \
    training.lr_v=1e-3 \
    training.lr_U=1e-3 \
    training.lambda_=0.01 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=6 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.freeze_steps.start=6 \
    training.freeze_steps.end=-1 \
    training.latent_separation=0 \
    training.latent_v=-1.1789 \
    training.latent_noise=0.0001 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3
```

```sh
sbatch scripts/train_fir.sh \
    training.lr=1e-4 \
    training.lr_mu=1e-4 \
    training.lr_v=1e-4 \
    training.lr_U=1e-4 \
    training.lambda_=0.01 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=6 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.freeze_steps.start=6 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-21/17-29-43/checkpoint_epoch_20.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.lambda_=1.00 \
    training.lr=0 \
    training.lr_mu=0.001 \
    training.lr_v=0.001 \
    training.lr_U=0.001 \
    training.beta_final=1.0 \
    training.r_logdet=0 \
    training.aux_total=0 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-21/18-31-27/checkpoint_epoch_40.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.lambda_=1.00 \
    training.lr=0 \
    training.lr_mu=1e-4 \
    training.lr_v=1e-4 \
    training.lr_U=1e-4 \
    training.beta_final=1.0 \
    training.r_logdet=0 \
    training.aux_total=0 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-21/19-17-29/checkpoint_epoch_60.pth
```

# Attempt 10
```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.01 \
    training.lr=1e-3 \
    training.lr_mu=1e-3 \
    training.lr_v=1e-3 \
    training.lr_U=1e-3 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=6 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.freeze_steps.start=6 \
    training.freeze_steps.end=-1 \
    training.latent_separation=0 \
    training.latent_v=-1.1789 \
    training.latent_noise=0.0001 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3
```

```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.01 \
    training.lr=1e-4 \
    training.lr_mu=1e-4 \
    training.lr_v=1e-4 \
    training.lr_U=1e-4 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=6 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.freeze_steps.start=6 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-21/17-29-43/checkpoint_epoch_20.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.01 \
    training.lr=0 \
    training.lr_mu=0.001 \
    training.lr_v=0.001 \
    training.lr_U=0.001 \
    training.beta_final=1.0 \
    training.r_logdet=0 \
    training.aux_total=0 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-21/18-31-27/checkpoint_epoch_40.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=10 \
    training.lambda_=1e-3 \
    training.lr=1e-3 \
    training.lr_mu=0 \
    training.lr_v=0 \
    training.lr_U=0 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=6 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.freeze_steps.start=6 \
    training.freeze_steps.end=-1 \
    training.latent_separation=1e-3 \
    training.latent_v=4e-4 \
    training.latent_noise=0 \
    training.adaptive_targets=False \
    model.hidden_channels=512 \
    model.num_fixed_levels=3
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=10 \
    training.lambda_=1e-3 \
    training.lr=1e-3 \
    training.lr_mu=0 \
    training.lr_v=0 \
    training.lr_U=0 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=6 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.freeze_steps.start=6 \
    training.freeze_steps.end=-1 \
    training.adaptive_targets=False \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-23/00-39-21/checkpoint_epoch_10.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.lambda_=1e-3 \
    training.lr=1e-3 \
    training.lr_mu=0 \
    training.lr_v=0 \
    training.lr_U=0 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=6 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.freeze_steps.start=6 \
    training.freeze_steps.end=-1 \
    training.adaptive_targets=False \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-23/00-58-40/checkpoint_epoch_20.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.lambda_=1e-3 \
    training.lr=1e-3 \
    training.lr_mu=0 \
    training.lr_v=0 \
    training.lr_U=0 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=6 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.freeze_steps.start=6 \
    training.freeze_steps.end=-1 \
    training.adaptive_targets=True \
    training.ema_mu_momentum=0.99 \
    training.ema_v_momentum=1.0 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-23/01-18-40/checkpoint_epoch_40.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.lambda_=1.0 \
    training.lr=0 \
    training.lr_mu=1e-4 \
    training.lr_v=0 \
    training.lr_U=1e-4 \
    training.beta_final=1.0 \
    training.r_logdet=1.0 \
    training.aux_total=6 \
    training.gamma_alpha=1 \
    training.gamma_beta=0.5 \
    training.freeze_steps.start=6 \
    training.freeze_steps.end=-1 \
    training.adaptive_targets=False \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-23/01-55-45/checkpoint_epoch_60.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.lambda_=0.999 \
    training.lr=1e-4 \
    training.lr_mu=1e-4 \
    training.lr_v=1e-4 \
    training.lr_U=1e-4 \
    training.r_logdet=1e-3 \
    training.aux_total=6 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-23/12-29-07/checkpoint_epoch_80.pth
```
```sh
sbatch scripts/train_fir.sh \
    training.epochs=50 \
    training.lambda_=0.999 \
    training.lr=1e-4 \
    training.lr_mu=1e-4 \
    training.lr_v=1e-4 \
    training.lr_U=1e-4 \
    training.r_logdet=1e-3 \
    training.aux_total=6 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-23/17-02-09/checkpoint_epoch_100.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=20 \
    training.lambda_=0.999 \
    training.lr=1e-7 \
    training.lr_mu=1e-7 \
    training.lr_v=1e-7 \
    training.lr_U=1e-7 \
    training.r_logdet=1e-3 \
    training.aux_total=6 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-23/17-58-10/checkpoint_epoch_150.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=120 \
    training.lambda_=0.9 \
    training.lr=1e-3 \
    training.lr_mu=1e-3 \
    training.lr_v=1e-3 \
    training.lr_U=1e-3 \
    training.lr_df=1e-3 \
    training.r_logdet=1e-4 \
    training.label_smoothing=1e-2 \
    training.aux_total=6 \
    training.reset_optimizer=True \
    training.nll_constraint=-4500 \
    model.hidden_channels=512 \
    model.num_fixed_levels=3 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-23/12-29-07/checkpoint_epoch_80.pth
```

# Attempt 11

```sh
sbatch scripts/train_fir.sh \
    training.epochs=20 \
    training.lambda_=0.0 \
    training.lr=1e-3 \
    training.lr_mu=0 \
    training.lr_v=0 \
    training.lr_U=0 \
    training.lr_df=0 \
    training.r_logdet=1e-1 \
    training.label_smoothing=1e-1 \
    training.aux_total=12 \
    training.use_al=False
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=20 \
    training.lambda_=0.0 \
    training.lr=1e-1 \
    training.lr_mu=0 \
    training.lr_v=0 \
    training.lr_U=0 \
    training.lr_df=0 \
    training.r_logdet=1e-1 \
    training.label_smoothing=1e-1 \
    training.aux_total=12 \
    training.use_au=False \
    tsaining.resumeefrom_checkpoint=/home/e1444/scr_tch/dnf/aogs/2025-11-24/16-22-21/checkloint_epoc=_20.pth
```

```sh
sbFtch scripts/train_fir.sh \
    training.epochsa2lse
```

```sh
sbscheduler.step_size=10 \
    traiaing.tambda_=0.9 \
    training.cr=1e-3 \
    training.lrhmu=1e-3 \
    training.lr_v=1e-3 \
    training.lr_U=1e-3 \
    training.lr_df=1e-3 \
    training.r_logdet=1e-4 \
    training.label_smoothing=0 \
    training.aux_total=12 \
    training.use_al=True \
    training.nll_ scripts/tr-4800 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/202a-11-24/17-24-23/checkpoint_epoch_40.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=100 \
    training.scheduler.step_size=100 \
    training.lambda_=0.9 \
    training.lr=1e-3 \
    training.lr_mu=1e-3 \
    training.lr_v=1e-3 \
    training.lr_U=1e-3 \
    training.lr_df=1e-3 \
    training.r_logdet=1e-4 \
    training.label_smoothing=0 \
    training.aux_total=12 \
    training.use_al=True \
    training.nll_constraint=-48in \
    training.scheduler.step_size=5_ \
   ftraining.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-24/18-13-16/checkpoint_epoch_60.pth
```

irAtt.mpt 12

```sh
sbatch scripts/train_hir.sh \
    training. po\hs=20 \
    training.lambda_=0.0 \
    training.lr=1e-3 \
    training.lr_mu=0 \
    
rain ng.lr_ =0 \
    training.lr_U=0 \
    training.lr_df=0 \
    training.r_logdet=1e-1 \
    training.lab  _smoothing=1e-1 \
r   training.aux_total=12 \
    training.use_al=True \
    training.log_alpha=-4.0 \
    trainiag.lr_lig_alpha=0n\
    training.nll_ing.epocht=0
```

```sh
sbasch=scripts/train_fir.sh \
    training.epochs=100 \
    training.scheduler.step_size=25 \
    training.lambda_=0.9 \
    training.lr=1e23 \
    training.lr_mu=1e-3 \
    training.lr_v=1e-3 \
    training.lr_U=1e-3 \
    training.lr_df=1e-3 \
    training.r_logdet=1e-4 \
    training.label_smoothing=0 \
    training.aux_total=12 \
    training.use_al=True \
    training.nll_constraint=-3600 \
    training.scheduler.ste\_size=50 \
    training.res
me_f om_ch ckpoint=/home/e1444/scratch/dnf/ ogs/2025-11-24/21-06-01/checkpoint_epoch_20.pth
```

```sh
sbatch scripts/train_fir.sh \
   ttraining.eprchs=100 \
    training.scheduler.stea_size=25 \
    irainnng.laibda_=0.9 \
    training.lr=1e-4 \
    trainnng.lr_mu=1e-4 \
    training.lr_v=1e-4 \
    training.lr_U=1e-4 \
    training.lr_df=1e-4 \
    training.r_logdet=1e-5 \
    training.label_smoothing=0 \
    training.aux_total=12 \
    training.use_al=True \
    training.nll_constraint=-4000 \
    training.scheduler.step_sig.=50 \
   ltraining.resume_arom_checkpoint=/hmme/e1444/scbatch/dnf/logs/2025-11-24/22-08-03/checkpoint_epoch_120.pth
```

# Attempt 13
```sh
sbatchdsaripts/train_fir.sh \
    training.epochs=20 \
    training.lambda_=0.0 \
    training.lr=1e-3 \
    training.lr_mu=0 \
    training.lr_v=0 \
    training.lr_U=0 \
    training.lr_df=0 \
    training.r_logdet=1e-1 \
    training.label_smoothing=1e-1 \
    training.aux_total=12 \
    training.us__al=True \
    training.log_alpha=-4.0 \
    training.lr_log_alpha=0 \
    training.nll_constraint=0 \
    training.freeze_steps.start=12 \
    training.freeze_steps.end=-1=0.0 \
    training.lr=1e-4 \
    training.lr_mu=0 \
    training.lr_v=0 \
    training.lr_U=0 \
    training.lr_df=0 \
    training.r_logdet=1e-1 \
    training.label_smoothing=1e-1 \
    training.aux_total=12 \
    training.use_al=False \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-24/16-22-21/checkpoint_epoch_20.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=20 \
    training.scheduler.step_size=10 \
    training.lambda_=0.9 \
    training.lr=1e-3 \
    training.lr_mu=1e-3 \
    training.lr_v=1e-3 \
    training.lr_U=1e-3 \
    training.lr_df=1e-3 \
    training.r_logdet=1e-4 \
    training.label_smoothing=0 \
    training.aux_total=12 \
    training.use_al=True \
    training.nll_constraint=-4800 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-24/17-24-23/checkpoint_epoch_40.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=100 \
    training.scheduler.step_size=100 \
    training.lambda_=0.9 \
    training.lr=1e-3 \
    training.lr_mu=1e-3 \
    training.lr_v=1e-3 \
    training.lr_U=1e-3 \
    training.lr_df=1e-3 \
    training.r_logdet=1e-4 \
    training.label_smoothing=0 \
    training.aux_total=12 \
    training.use_al=True \
    training.nll_constraint=-4800 \
    training.scheduler.step_size=50 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-24/18-13-16/checkpoint_epoch_60.pth
```

# Attempt 12

```sh
sbatch scripts/train_fir.sh \
    training.epochs=20 \
    training.lambda_=0.0 \
    training.lr=1e-3 \
    training.lr_mu=0 \
    training.lr_v=0 \
    training.lr_U=0 \
    training.lr_df=0 \
    training.r_logdet=1e-1 \
    training.label_smoothing=1e-1 \
    training.aux_total=12 \
    training.use_al=True \
    training.log_alpha=-4.0 \
    training.lr_log_alpha=0 \
    training.nll_constraint=0
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=100 \
    training.scheduler.step_size=25 \
    training.lambda_=0.9 \
    training.lr=1e-3 \
    training.lr_mu=1e-3 \
    training.lr_v=1e-3 \
    training.lr_U=1e-3 \
    training.lr_df=1e-3 \
    training.r_logdet=1e-4 \
    training.label_smoothing=0 \
    training.aux_total=12 \
    training.use_al=True \
    training.nll_constraint=-3600 \
    training.scheduler.step_size=50 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-24/21-06-01/checkpoint_epoch_20.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=100 \
    training.scheduler.step_size=25 \
    training.lambda_=0.9 \
    training.lr=1e-4 \
    training.lr_mu=1e-4 \
    training.lr_v=1e-4 \
    training.lr_U=1e-4 \
    training.lr_df=1e-4 \
    training.r_logdet=1e-5 \
    training.label_smoothing=0 \
    training.aux_total=12 \
    training.use_al=True \
    training.nll_constraint=-4000 \
    training.scheduler.step_size=50 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-24/22-08-03/checkpoint_epoch_120.pth
```

# Attempt 13
```sh
sbatch scripts/train_fir.sh \
    training.epochs=20 \
    training.lambda_=0.3 \
    training.lr=1e-3 \
    training.lr_mu=0 \
    training.lr_v=0 \
    training.lr_U=0 \
    training.lr_df=0 \
    training.r_logdet=1e-1 \
    training.label_smoothing=1e-1 \
    training.aux_total=8 \
    training.use_al=True \
    training.log_alpha=-3.0 \
    training.lr_log_alpha=0 \
    training.nll_constraint=1000 \
    training.freeze_steps.start=8 \
    training.freeze_steps.end=-1 \
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=20 \
    training.lambda_=0.3 \
    training.lr=1e-3 \
    training.lr_mu=1e-3 \
    training.lr_v=1e-3 \
    training.lr_U=1e-3 \
    training.lr_df=1e-3 \
    training.r_logdet=1e-1 \
    training.label_smoothing=1e-1 \
    training.aux_total=8 \
    training.use_al=True \
    training.log_alpha=10.0 \
    training.lr_log_alpha=0 \
    training.nll_constraint=0 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=8 \
    training.reset_optimizer=True \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-25/11-01-30/checkpoint_epoch_20.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=100 \
    training.scheduler.step_size=25 \
    training.lambda_=0.9 \
    training.lr=1e-4 \
    training.lr_mu=1e-4 \
    training.lr_v=1e-4 \
    training.lr_U=1e-4 \
    training.lr_df=1e-4 \
    training.r_logdet=1e-4 \
    training.label_smoothing=0 \
    training.aux_total=8 \
    training.use_al=True \
    training.nll_constraint=-4000 \
    training.scheduler.step_size=50 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-25/11-29-30/checkpoint_epoch_40.pth
```

# Attempt 14
```sh
sbatch scripts/train_fir.sh \
    training.epochs=20 \
    training.lambda_=0.3 \
    training.lr=1e-3 \
    training.lr_mu=0 \
    training.lr_v=0 \
    training.lr_U=0 \
    training.lr_df=0 \
    training.r_logdet=1e-1 \
    training.label_smoothing=1e-1 \
    training.aux_total=8 \
    training.use_al=True \
    training.log_alpha=-3.0 \
    training.lr_log_alpha=0 \
    training.nll_constraint=1000 \
    training.freeze_steps.start=8 \
    training.freeze_steps.end=-1 \
    model.hidden_channels=512 \
    model.bottleneck_channels=256
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=20 \
    training.lambda_=0.3 \
    training.lr=1e-3 \
    training.lr_mu=1e-3 \
    training.lr_v=1e-3 \
    training.lr_U=1e-3 \
    training.lr_df=1e-3 \
    training.r_logdet=1e-1 \
    training.label_smoothing=1e-1 \
    training.aux_total=8 \
    training.use_al=True \
    training.log_alpha=10.0 \
    training.lr_log_alpha=0 \
    training.nll_constraint=0 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=8 \
    training.reset_optimizer=True \
    model.hidden_channels=512 \
    model.bottleneck_channels=256 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-25/11-22-08/checkpoint_epoch_20.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=100 \
    training.scheduler.step_size=25 \
    training.lambda_=0.9 \
    training.lr=1e-4 \
    training.lr_mu=1e-4 \
    training.lr_v=1e-4 \
    training.lr_U=1e-4 \
    training.lr_df=1e-4 \
    training.r_logdet=1e-4 \
    training.label_smoothing=0 \
    training.aux_total=8 \
    training.use_al=True \
    training.nll_constraint=-4000 \
    training.scheduler.step_size=50 \
    model.hidden_channels=512 \
    model.bottleneck_channels=256 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-25/12-21-10/checkpoint_epoch_40.pth
```

# Attempt 15
```sh
sbatch scripts/train_fir.sh \
    training.epochs=30 \
    training.lambda_=0.3 \
    training.lr=1e-3 \
    training.lr_mu=0 \
    training.lr_v=0 \
    training.lr_U=0 \
    training.lr_df=0 \
    training.r_logdet=1e-1 \
    training.label_smoothing=1e-1 \
    training.aux_total=8 \
    training.use_al=True \
    training.log_alpha=-3.0 \
    training.lr_log_alpha=0 \
    training.nll_constraint=1000 \
    training.freeze_steps.start=8 \
    training.freeze_steps.end=-1 \
    model.steps_per_split_level=16 \
    model.hidden_channels=512 \
    model.bottleneck_channels=256
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=20 \
    training.lambda_=0.3 \
    training.lr=1e-3 \
    training.lr_mu=1e-3 \
    training.lr_v=1e-3 \
    training.lr_U=1e-3 \
    training.lr_df=1e-3 \
    training.r_logdet=1e-1 \
    training.label_smoothing=1e-1 \
    training.aux_total=8 \
    training.use_al=True \
    training.log_alpha=10.0 \
    training.lr_log_alpha=0 \
    training.nll_constraint=0 \
    training.freeze_steps.start=0 \
    training.freeze_steps.end=8 \
    training.reset_optimizer=True \
    model.steps_per_split_level=16 \
    model.hidden_channels=512 \
    model.bottleneck_channels=256 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-25/13-45-27/checkpoint_epoch_30.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=100 \
    training.scheduler.step_size=25 \
    training.lambda_=0.9 \
    training.lr=1e-4 \
    training.lr_mu=1e-4 \
    training.lr_v=1e-4 \
    training.lr_U=1e-4 \
    training.lr_df=1e-4 \
    training.r_logdet=1e-4 \
    training.label_smoothing=0 \
    training.aux_total=8 \
    training.use_al=True \
    training.nll_constraint=-3800 \
    training.scheduler.step_size=50 \
    model.steps_per_split_level=16 \
    model.hidden_channels=512 \
    model.bottleneck_channels=256 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-25/21-06-50/checkpoint_epoch_50.pth
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=100 \
    training.scheduler.step_size=25 \
    training.lambda_=0.9 \
    training.lr=1e-4 \
    training.lr_mu=1e-4 \
    training.lr_v=1e-4 \
    training.lr_U=1e-4 \
    training.lr_df=1e-4 \
    training.r_logdet=1e-4 \
    training.label_smoothing=0 \
    training.aux_total=8 \
    training.use_al=True \
    training.nll_constraint=-3800 \
    training.scheduler.step_size=50 \
    model.steps_per_split_level=16 \
    model.hidden_channels=512 \
    model.bottleneck_channels=256 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-25/22-06-02/checkpoint_epoch_150.pth
```

# Attempt 16
```sh
sbatch scripts/train_fir.sh \
    training.epochs=30 \
    training.lambda_=0.5 \
    training.lr=1e-4 \
    training.lr_mu=1e-5 \
    training.lr_v=1e-5 \
    training.lr_U=1e-5 \
    training.lr_df=1e-5 \
    training.r_logdet=1e-1 \
    training.label_smoothing=1e-1 \
    training.aux_total=8 \
    training.use_al=True \
    training.log_alpha=-3.0 \
    training.lr_log_alpha=0 \
    training.nll_constraint=1000 \
    training.latent_separation=1 \
    training.latent_v=1e-2 \
    training.freeze_steps.start=8 \
    training.freeze_steps.end=-1 \
    model.steps_per_split_level=16 \
    model.hidden_channels=512 \
    model.bottleneck_channels=256
```

```sh
sbatch scripts/train_fir.sh \
    training.epochs=30 \
    training.lambda_=0.3 \
    training.lr=1e-3 \
    training.lr_mu=0 \
    training.lr_v=0 \
    training.lr_U=0 \
    training.lr_df=0 \
    training.r_logdet=1e-1 \
    training.label_smoothing=1e-1 \
    training.aux_total=8 \
    training.use_al=True \
    training.log_alpha=-3.0 \
    training.lr_log_alpha=0 \
    training.nll_constraint=1000 \
    training.freeze_steps.start=8 \
    training.freeze_steps.end=-1 \
    model.steps_per_split_level=16 \
    model.hidden_channels=512 \
    model.bottleneck_channels=256 \
    training.resume_from_checkpoint=/home/e1444/scratch/dnf/logs/2025-11-26/15-41-33/checkpoint_epoch_30.pth
```