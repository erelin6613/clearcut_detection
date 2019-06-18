import sys

sys.path.append("../..")

import collections

import torch
from catalyst.dl.callbacks import DiceCallback, CheckpointCallback, InferCallback
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.utils import UtilsFactory

from datasets import create_loaders
from models.utils import get_model
from params import args
from torch.nn import CrossEntropyLoss


def main():
    model = get_model('fpn50_multiclass')

    print("Loading model")
    model, device = UtilsFactory.prepare_model(model)

    loaders = create_loaders()

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.3)

    # model runner
    runner = SupervisedRunner()

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[
            # DiceCallback()
            # EarlyStoppingCallback(patience=20, min_delta=0.01)
        ],
        logdir=args.logdir,
        num_epochs=args.epochs,
        verbose=True
    )

    infer_loader = collections.OrderedDict([("infer", loaders["valid"])])
    runner.infer(
        model=model,
        loaders=infer_loader,
        callbacks=[
            CheckpointCallback(
                resume=f"{args.logdir}/checkpoints/best.pth"),
            InferCallback()
        ],
    )


if __name__ == '__main__':
    main()
