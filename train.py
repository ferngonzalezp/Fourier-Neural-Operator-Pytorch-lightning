from argparse import ArgumentParser
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from fourier_neural_op_pl import FNO
from ns2d_dm import ns2d_dm

def main(hparams):
    checkpoint_callback = ModelCheckpoint(
    dirpath=os.getcwd(),
    save_top_k=True,
    save_last=True,
    verbose=True,
    monitor='val_loss',
    mode='min')
    model = FNO(hparams)
    dm = ns2d_dm(hparams.path,batch_size=hparams.batch_size)
    trainer = Trainer.from_argparse_args(hparams,callbacks=checkpoint_callback,
                         auto_select_gpus = True,
                         precision = 32,
                         log_every_n_steps=10,
                         progress_bar_refresh_rate=1)  
    trainer.fit(model, datamodule = dm)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--path', type=str)
    parser.add_argument('--batch_size', type=int)
    FNO.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)