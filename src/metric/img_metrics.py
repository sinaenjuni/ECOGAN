import torch
from metric.inception_net_V2 import EvalModel

from metric.ins import calculate_kl_div
from metric.fid import calculate_mu_sigma, frechet_inception_distance


class Fid_and_is(torch.nn.Module):
    def __init__(self):
        super(Fid_and_is, self).__init__()
        self.eval_model = EvalModel()
        self.eval_model.eval()

        self.real_feature = []
        self.real_logit = []
        self.fake_feature = []
        self.fake_logit = []


    def update(self, x, real):
        with torch.no_grad():
            feature, logit = self.eval_model.get_outputs(x, quantize=True)
            logit = torch.nn.functional.softmax(logit, dim=1)
            if real:
                self.real_feature.append(feature)
                self.real_logit.append(logit)
            else:
                self.fake_feature.append(feature)
                self.fake_logit.append(logit)


    def compute_ins(self):
        fake_logit = torch.cat(self.fake_logit)
        return calculate_kl_div(fake_logit, 10)

    def compute_fid(self):
        real_feature = torch.cat(self.real_feature)
        fake_feature = torch.cat(self.fake_feature)
        mu_real, sigma_real = calculate_mu_sigma(real_feature.cpu().numpy())
        mu_fake, sigma_fake = calculate_mu_sigma(fake_feature.cpu().numpy())
        fid_score = frechet_inception_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        return fid_score

    def reset(self, real):
        if real:
            self.real_feature = []
            self.real_logit = []
        else:
            self.fake_feature = []
            self.fake_logit = []

# inputs = torch.rand(100, 3, 32, 32).cuda()
# metric = Fid_and_is().cuda()
# with torch.no_grad():
#     featrue, logit = metric.eval_model.get_outputs(inputs)
# torch.nn.functional.softmax(logit, dim=1)
# [p for p in metric.eval_model.parameters()]
