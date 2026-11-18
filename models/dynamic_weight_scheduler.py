class DynamicLossWeightScheduler:
    """TVRL dynamic loss weight scheduling during the warm-up period."""
    def __init__(self, args, total_epochs):
        self.alpha_initial = 0.0
        self.alpha_target = getattr(args, 'alpha', 1e-2)

        self.beta_initial = 0.1
        self.beta_target = getattr(args, 'beta', 1.0)

        self.gamma_initial = 0.0
        self.gamma_target = getattr(args, 'gamma', 1e-2)

        self.lambda_t_initial = 0.05
        self.lambda_t_target = getattr(args, 'lambda_t', getattr(args, 'trust_loss_weight', 0.2))

        self.uncertainty_loss_weight_initial = 0.02
        self.uncertainty_loss_weight_target = getattr(args, 'uncertainty_loss_weight', 0.1)

        self.warmup_ratio = 0.1
        self.warmup_epochs = max(1, int(total_epochs * self.warmup_ratio))

    def get_weights(self, current_epoch):
        if current_epoch < self.warmup_epochs:
            progress = current_epoch / self.warmup_epochs
            alpha = self.alpha_initial + (self.alpha_target - self.alpha_initial) * progress
            beta = self.beta_initial + (self.beta_target - self.beta_initial) * progress
            gamma = self.gamma_initial + (self.gamma_target - self.gamma_initial) * progress
            lambda_t = self.lambda_t_initial + (self.lambda_t_target - self.lambda_t_initial) * progress
            uncertainty_loss_weight = self.uncertainty_loss_weight_initial + (self.uncertainty_loss_weight_target - self.uncertainty_loss_weight_initial) * progress
        else:
            alpha = self.alpha_target
            beta = self.beta_target
            gamma = self.gamma_target
            lambda_t = self.lambda_t_target
            uncertainty_loss_weight = self.uncertainty_loss_weight_target

        return {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'lambda_t': lambda_t,
            'uncertainty_loss_weight': uncertainty_loss_weight,
        }
