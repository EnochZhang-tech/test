
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FilterMSELoss", "FilterHuberLoss", "LogLoss", "Ada_Mse"]


class FilterMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(FilterMSELoss, self).__init__()
        print('FilterMSELoss')

    def forward(self, pred, gold):
        return torch.mean(F.mse_loss(pred, gold, reduction='none'))


class FilterHuberLoss(nn.Module):
    def __init__(self, delta=5, **kwargs):
        super(FilterHuberLoss, self).__init__()
        self.delta = delta #超参数
        print('FilterHuberLoss', 'delta = {}'.format(self.delta))

    def forward(self, pred, gold):
        return torch.mean(F.smooth_l1_loss(pred, gold, reduction='none', beta=self.delta))

class LogLoss(torch.nn.Module):
    def __init__(self, epsilon=0.5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, true, pred):
        true = torch.log(true + self.epsilon)
        pred = torch.log(pred + self.epsilon)

        loss = torch.mean(torch.abs(pred - true))
        return loss

class Ada_Mse(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, true, pred):
        beta = torch.exp(-10 * true + 3) + 1
        mse = torch.square(true - pred)
        loss = torch.mean(beta * mse)
        return loss

class MMD(torch.nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type
        self.flatten = torch.nn.Flatten()

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        source = self.flatten(source)
        target = self.flatten(target)
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            # torch.cuda.empty_cache()
            return loss

class CORAL(torch.nn.Module):
    def __init__(self, in_dim):
        super(CORAL, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=(1, 1))
        self.flaten = torch.nn.Flatten()

    def forward(self, source, target):
        source = self.conv2d(source)
        target = self.conv2d(target)
        source = self.flaten(source)
        target = self.flaten(target)
        d = source.size(1)

        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        # loss = loss / (4 * d * d)
        return loss

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)

class VAT(torch.nn.Module):
    def __init__(self, model, predictor, device):
        super(VAT, self).__init__()
        self.n_power = 1
        self.XI = 1e-6
        self.feature_extractor = model
        self.predictor = predictor
        self.epsilon = 3.5
        self.device = device

    def forward(self, X, logit, lap):
        vat_loss = self.virtual_adversarial_loss(X, logit, lap)
        return vat_loss

    def generate_virtual_adversarial_perturbation(self, x, logit, lap):
        d = torch.randn_like(x, device=self.device)

        for _ in range(self.n_power):
            d = self.XI * self.get_normalized_vector(d).requires_grad_()
            logit_m, _ = self.feature_extractor(x + d, lap)
            logit_m = self.predictor(logit_m)
            dist = self.kl_divergence_with_logit(logit, logit_m)
            grad = torch.autograd.grad(dist, [d])[0]
            d = grad.detach()

        return self.epsilon * self.get_normalized_vector(d)

    def kl_divergence_with_logit(self, q_logit, p_logit):
        q = F.softmax(q_logit, dim=1)
        qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
        qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
        return qlogq - qlogp

    def get_normalized_vector(self, d):
        return F.normalize(d.view(d.size(0), -1), p=2, dim=1).reshape(d.size())

    def virtual_adversarial_loss(self, x, logit, lap):
        r_vadv = self.generate_virtual_adversarial_perturbation(x, logit, lap)
        logit_p = logit.detach()
        logit_m, _ = self.feature_extractor(x + r_vadv, lap)
        logit_m = self.predictor(logit_m)
        loss = self.kl_divergence_with_logit(logit_p, logit_m)
        return loss