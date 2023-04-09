# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
# +
import IPython.display as ipd
import pytorch_lightning as pl
import torch

from mridangam.data import MridangamDataModule
from mridangam.loss import MSS
from mridangam.loss import StationaryRegularization
from mridangam.loss import TransientRegularization
from mridangam.models import MLP
from mridangam.models import TCN
from mridangam.tasks import MridangamTonicClassification
from mridangam.tasks import TransientStationarySeparation

# %load_ext autoreload
# %autoreload 2
# -

# # Mridangam Tonic Classification
#
# For this first experiment I am using embeddings generated by Crepe with a simple
# deep MLP classifier.

# The first step is to preprocess our dataset of audio files and create a PyTorch
# Lightning DataModule for our experiment. In this step we compute embeddings using
# the full Crepe model and save the result embeddings to disk.

datamodule = MridangamDataModule(
    dataset_dir="dataset/preprocesed",
    unprocessed_dir="dataset/mridangam_stroke_1.5/",
    batch_size=1,
    num_workers=0,
    attribute="tonic",
    max_files=500,
)
datamodule.prepare_data()

datamodule.setup("fit")
train_dataloader = datamodule.train_dataloader()
val_dataloader = datamodule.val_dataloader()

# +
# Get input feature size and target num_classes from data
audio, embedding, label = next(iter(train_dataloader))

in_features = embedding.size(-1)
print(in_features)

out_features = train_dataloader.dataset.num_classes
print(out_features)

print(label.dtype)
# -

mlp = MLP(in_features=in_features, hidden=[256, 128], out_features=out_features)
model = MridangamTonicClassification(model=mlp)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
trainer = pl.Trainer(max_epochs=1000, accelerator=accelerator, log_every_n_steps=10)
trainer.fit(
    model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)

datamodule.setup("test")
test_dataloader = datamodule.test_dataloader()
trainer.test(model=model, dataloaders=test_dataloader)

# +
print(len(test_dataloader.dataset))
audio, emb, label = test_dataloader.dataset[9]

print(label)
y = mlp(emb)
print(torch.argmax(y))
# -

# # Mridangam Transient/Stationary Separation

# +
transient_tcn = TCN(
    in_channels=1,
    hidden_channels=16,
    out_channels=1,
    dilation_base=2,
    num_layers=12,
    kernel_size=3,
)

sustain_tcn = TCN(
    in_channels=1,
    hidden_channels=16,
    out_channels=1,
    dilation_base=2,
    num_layers=12,
    kernel_size=3,
)

# +
transient_loss = TransientRegularization()
stationary_loss = StationaryRegularization()
recon_loss = MSS()

tss_model = TransientStationarySeparation(
    transient_tcn,
    sustain_tcn,
    reconstruction_loss=recon_loss,
    transient_loss=transient_loss,
    stationary_loss=stationary_loss,
)

# +
audio, embedding, label = next(iter(train_dataloader))

trans, sus = tss_model(audio)
print(trans.shape)
print(sus.shape)
# -

trainer = pl.Trainer(max_epochs=1000, accelerator="cpu")
trainer.fit(model=tss_model, train_dataloaders=train_dataloader)

# +
trans, sus = tss_model(audio)

ipd.display(ipd.Audio(audio[0, 0].detach().cpu().numpy(), rate=48000))
ipd.display(ipd.Audio(trans[0, 0].detach().cpu().numpy(), rate=48000))
ipd.display(ipd.Audio(sus[0, 0].detach().cpu().numpy(), rate=48000))
# -
