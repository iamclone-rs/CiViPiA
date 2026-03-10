import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import retrieval_average_precision
try:
    import pytorch_lightning as pl
except ImportError:
    import lightning.pytorch as pl

from src.clip import clip
from experiments.options import opts

def unfreeze_visual_layer_norms(model):
    for module in model.visual.modules():
        if isinstance(module, torch.nn.LayerNorm):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

class Model(pl.LightningModule):
    def __init__(self, seen_categories):
        super().__init__()

        self.opts = opts
        self.seen_categories = sorted(seen_categories)
        self.save_hyperparameters({'seen_categories': self.seen_categories})
        self.category_to_idx = {
            category: idx for idx, category in enumerate(self.seen_categories)}
        self.clip, _ = clip.load('ViT-B/32', device=self.device)
        self.clip.requires_grad_(False)
        unfreeze_visual_layer_norms(self.clip)

        # Prompt Engineering
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=0.2)
        self.cls_loss_fn = nn.CrossEntropyLoss()
        text_prompts = [
            'a photo of a {}'.format(category.replace('_', ' '))
            for category in self.seen_categories]
        self.register_buffer(
            'text_tokens',
            clip.tokenize(text_prompts, truncate=True),
            persistent=False)

        self.best_metric = -1e3
        self.validation_step_outputs = []

    def configure_optimizers(self):
        clip_trainable_parameters = [
            parameter for parameter in self.clip.parameters()
            if parameter.requires_grad]
        optimizer = torch.optim.Adam([
            {'params': clip_trainable_parameters, 'lr': self.opts.clip_LN_lr},
            {'params': [self.sk_prompt, self.img_prompt], 'lr': self.opts.prompt_lr}])
        return optimizer

    def encode_text_features(self):
        with torch.no_grad():
            text_features = self.clip.encode_text(self.text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def compute_classification_loss(self, features, labels):
        text_features = self.encode_text_features()
        logits = self.clip.logit_scale.exp().detach() * (
            F.normalize(features, dim=-1) @ text_features.t())
        return self.cls_loss_fn(logits, labels)

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            feat = self.clip.encode_image(
                data, self.img_prompt.expand(data.shape[0], -1, -1))
        else:
            feat = self.clip.encode_image(
                data, self.sk_prompt.expand(data.shape[0], -1, -1))
        return feat

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        labels = torch.tensor(
            [self.category_to_idx[cat] for cat in category],
            device=self.device,
            dtype=torch.long)
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        sk_cls_loss = self.compute_classification_loss(sk_feat, labels)
        img_cls_loss = self.compute_classification_loss(img_feat, labels)
        cls_loss = sk_cls_loss + img_cls_loss
        loss = triplet_loss + self.opts.cls_loss_weight * cls_loss
        self.log('train_triplet_loss', triplet_loss, on_step=True, on_epoch=True, batch_size=sk_tensor.shape[0])
        self.log('train_cls_loss', cls_loss, on_step=True, on_epoch=True, batch_size=sk_tensor.shape[0])
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=sk_tensor.shape[0])
        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=sk_tensor.shape[0])
        self.validation_step_outputs.append({
            'sketch_feat': sk_feat.detach().cpu(),
            'image_feat': img_feat.detach().cpu(),
            'category': list(category),
        })
        return loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        query_feat_all = torch.cat([output['sketch_feat'] for output in self.validation_step_outputs])
        gallery_feat_all = torch.cat([output['image_feat'] for output in self.validation_step_outputs])
        all_category = np.array(sum([output['category'] for output in self.validation_step_outputs], []))


        ## mAP category-level SBIR Metrics
        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all))
        for idx, sk_feat in enumerate(query_feat_all):
            category = all_category[idx]
            distance = -1*self.distance_fn(sk_feat.unsqueeze(0), gallery)
            target = torch.zeros(len(gallery), dtype=torch.bool)
            target[np.where(all_category == category)] = True
            ap[idx] = retrieval_average_precision(distance.cpu(), target.cpu())
        
        mAP = torch.mean(ap)
        self.log('mAP', mAP, prog_bar=True)
        if self.global_step > 0:
            self.best_metric = self.best_metric if  (self.best_metric > mAP.item()) else mAP.item()
        print ('mAP: {}, Best mAP: {}'.format(mAP.item(), self.best_metric))
        self.validation_step_outputs.clear()
