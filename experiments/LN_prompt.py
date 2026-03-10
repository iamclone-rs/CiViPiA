import os
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
try:
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
except ImportError:
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

from src.model_LN_prompt import Model
from src.dataset_retrieval import Sketchy
from experiments.options import opts


class EpochMetricsPrinter(Callback):
    @staticmethod
    def _format_metric(metrics, name):
        value = metrics.get(name)
        if value is None:
            return None
        if hasattr(value, 'item'):
            value = value.item()
        return '{}={:.4f}'.format(name, value)

    def _print_metrics(self, trainer, include_validation_metrics):
        metrics = trainer.callback_metrics
        metric_names = [
            'train_loss',
            'train_triplet_loss',
            'train_cls_loss',
        ]
        if include_validation_metrics:
            metric_names.extend(['val_loss', 'mAP'])
        formatted_metrics = [
            self._format_metric(metrics, name)
            for name in metric_names]
        formatted_metrics = [metric for metric in formatted_metrics if metric is not None]
        epoch = trainer.current_epoch + 1
        print('epoch {:03d}: {}'.format(epoch, ' | '.join(formatted_metrics)))

    def on_train_epoch_end(self, trainer, pl_module):
        check_val_every_n_epoch = trainer.check_val_every_n_epoch or 1
        if trainer.val_dataloaders is not None and (trainer.current_epoch + 1) % check_val_every_n_epoch == 0:
            return
        self._print_metrics(trainer, include_validation_metrics=False)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._print_metrics(trainer, include_validation_metrics=True)

if __name__ == '__main__':
    dataset_transforms = Sketchy.data_transform(opts)

    train_dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
    val_dataset = Sketchy(opts, dataset_transforms, mode='val', used_cat=train_dataset.all_categories, return_orig=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers)

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='saved_models/%s'%opts.exp_name,
        filename='{epoch:02d}-{val_loss:.4f}',
        mode='min',
        save_last=True)

    ckpt_path = os.path.join('saved_models', opts.exp_name, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None

    trainer = Trainer(
        accelerator='auto',
        devices='auto',
        min_epochs=1, max_epochs=60,
        benchmark=True,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=logger,
        num_sanity_val_steps=0,
        # val_check_interval=10, 
        # accumulate_grad_batches=1,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, EpochMetricsPrinter()]
    )

    model = Model(seen_categories=train_dataset.all_categories)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_path)
