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
    training.r_logdet=1e-4 \
    training.label_smoothing=1e-1 \
    training.aux_total=12 \
    training.lr_alpha=0 \
    training.nll_constraint=5000 # effectively no constraint -> purely optimize for ce
```