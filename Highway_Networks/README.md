# Highway Networks on CIFAR-10

## Setup

Run from the `Deep_Learning` folder (Python 3.12, CUDA 12.6):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install matplotlib
```

Download CIFAR-10 (python version) from https://www.cs.toronto.edu/~kriz/cifar.html and extract it to `Highway_Networks/data/cifar-10-batches-py`.

## Run

Run from the `Highway_Networks` folder:

```powershell
..\.venv\Scripts\python.exe train.py --model highway --blocks 10
..\.venv\Scripts\python.exe train.py --model plain   --blocks 10
..\.venv\Scripts\python.exe train.py --model highway --blocks 20
..\.venv\Scripts\python.exe train.py --model plain   --blocks 20
```

Other options: `--channels`, `--epochs` (default 50), `--batch-size`, `--lr`, `--seed`.

## Files

```
Highway_Networks/
├── models.py      # HighwayBlock, PlainBlock, CIFAR10Net
├── train.py       # training, validation, test, plots
├── data/cifar-10-batches-py/
└── runs/<model>_<blocks>_blocks_lr<lr>/
    ├── results.json          # settings, best validation and test results
    ├── best_model.pt         # best validation checkpoint
    ├── history.csv           # loss, accuracy, gate activity per epoch
    ├── training_curves.png   # learning curves
    └── gate_means.png        # mean T(x) per block (highway only)
```
