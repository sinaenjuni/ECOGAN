import torch
import numpy as np

class ExhustiveContrastiveLoss(torch.nn.Module):
    def __init__(self, num_classes, temperature):
        super(ExhustiveContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _make_neg_removal_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        n_samples = labels.shape[0]
        mask_multi, target = np.zeros([self.num_classes, n_samples]), 1.0
        for c in range(self.num_classes):
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long)

    # def make_index_matrix(self, labels):
    #     labels = labels.detach().cpu().numpy()
    #     num_samples = labels.shape[0]
    #     mask_multi, target = np.ones([self.num_classes, num_samples]), 0.0
    #
    #     for c in range(self.num_classes):
    #         c_indices = np.where(labels==c)
    #         mask_multi[c, c_indices] = target
    #     return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, embed_data, embed_label, label, **_):
        device = torch.device(embed_data.device)
        f2f_sim = self.calculate_similarity_matrix(embed_data, embed_data)
        f2f_sim = self._remove_diag(f2f_sim)
        f2f_max, _ = torch.max(f2f_sim, dim=1, keepdim=True)

        f2f_logits = f2f_sim - f2f_max.detach()
        f2f_logits = torch.exp(f2f_logits / self.temperature)

        pos_mask_redia = self._remove_diag(self._make_neg_removal_mask(label)[label]).to(device)
        f2f_logits_pos_only = pos_mask_redia * f2f_logits

        # emb2proxy = torch.exp(self.cosine_similarity(embed_data, embed_label) / self.temperature)

        e2p_sim = self.calculate_similarity_matrix(embed_data, embed_label)
        e2p_max, _ = torch.max(e2p_sim, dim=1, keepdim=True)

        e2p_logits = e2p_sim - e2p_max.detach()
        e2p_logits = torch.exp(e2p_logits / self.temperature)

        pos_mask = self._make_neg_removal_mask(label)[label].to(device)
        e2p_logits_pos_only = pos_mask * e2p_logits


        # numerator = emb2proxy + sim_pos_only.sum(dim=1)
        numerator = e2p_logits_pos_only.sum(dim=1) + f2f_logits_pos_only.sum(dim=1)
        denomerator = e2p_logits.sum(dim=1) + f2f_logits.sum(dim=1)
        return -torch.log(numerator / denomerator).mean()


class ConditionalContrastiveLoss(torch.nn.Module):
    def __init__(self, num_classes, temperature):
        super(ConditionalContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _make_neg_removal_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        n_samples = labels.shape[0]
        mask_multi, target = np.zeros([self.num_classes, n_samples]), 1.0
        for c in range(self.num_classes):
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, embed_data, embed_label, label, **_):
        device = torch.device(embed_data.device)
        sim_matrix = self.calculate_similarity_matrix(embed_data, embed_data)
        sim_matrix = torch.exp(self._remove_diag(sim_matrix) / self.temperature)
        neg_removal_mask = self._remove_diag(self._make_neg_removal_mask(label)[label]).to(device)
        sim_pos_only = neg_removal_mask * sim_matrix

        emb2proxy = torch.exp(self.cosine_similarity(embed_data, embed_label) / self.temperature)

        numerator = emb2proxy + sim_pos_only.sum(dim=1)
        denomerator = torch.cat([torch.unsqueeze(emb2proxy, dim=1), sim_matrix], dim=1).sum(dim=1)
        return -torch.log(numerator / denomerator).mean()