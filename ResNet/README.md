# ResNet on CIFAR-10

## Setup

Run from the `Deep_Learning` folder (Python 3.12, CUDA 12.6):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install matplotlib
```

Download CIFAR-10 (python version) from https://www.cs.toronto.edu/~kriz/cifar.html and extract it to `ResNet/data/cifar-10-batches-py`.

## Run

Run from the `ResNet` folder:

```powershell
..\.venv\Scripts\python.exe train.py --model resnet --depth 56
..\.venv\Scripts\python.exe train.py --model plain  --depth 56
..\.venv\Scripts\python.exe train.py --model resnet --depth 20
..\.venv\Scripts\python.exe train.py --model plain  --depth 20
```

Other options: `--iterations` (default 64000), `--batch-size`, `--lr`, `--warmup-iters`, `--val-split`, `--seed`.

## Files

```
ResNet/
├── models.py      # ShortcutA, BasicBlock, CIFARResNet
├── train.py       # training, evaluation, plots
├── analysis.ipynb # comparison of plain, highway and resnet runs
├── data/cifar-10-batches-py/
└── runs/<model>_<depth>/
    ├── results.json          # settings, final train and test error
    ├── model.pt              # model after the last iteration
    ├── history.csv           # loss, error per epoch
    └── training_curves.png   # learning curves
```
