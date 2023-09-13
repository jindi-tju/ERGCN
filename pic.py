import subprocess

cmds = ['python train.py --lr=2e-5 --dropout=0.2 --l2reg=0.00001 --seed=42',
        'python train.py --lr=2e-5 --dropout=0.2 --l2reg=0.00001 --seed=66',
        'python train.py --lr=2e-5 --dropout=0.2 --l2reg=0.00001 --seed=300',
        'python train.py --lr=2e-5 --dropout=0.1 --l2reg=0.00001 --seed=42',
        'python train.py --lr=2e-5 --dropout=0.1 --l2reg=0.00001 --seed=222',
        'python train.py --lr=2e-5 --dropout=0.1 --l2reg=0.00001 --seed=300',
        'python train.py --lr=1e-5 --dropout=0.2 --l2reg=0.00001 --seed=42',
        'python train.py --lr=1e-5 --dropout=0.2 --l2reg=0.00001 --seed=66',
        'python train.py --lr=1e-5 --dropout=0.2 --l2reg=0.00001 --seed=300',
        'python train.py --lr=1e-5 --dropout=0.1 --l2reg=0.00001 --seed=42',
        'python train.py --lr=1e-5 --dropout=0.1 --l2reg=0.00001 --seed=222',
        'python train.py --lr=1e-5 --dropout=0.1 --l2reg=0.00001 --seed=300',
        'python train.py --lr=5e-5 --dropout=0.2 --l2reg=0.00001 --seed=42',
        'python train.py --lr=5e-5 --dropout=0.2 --l2reg=0.00001 --seed=66',
        'python train.py --lr=5e-5 --dropout=0.2 --l2reg=0.00001 --seed=300',
        'python train.py --lr=5e-5 --dropout=0.1 --l2reg=0.00001 --seed=42',
        'python train.py --lr=5e-5 --dropout=0.1 --l2reg=0.00001 --seed=222',
        'python train.py --lr=5e-5 --dropout=0.1 --l2reg=0.00001 --seed=300',
        'python train.py --lr=2e-5 --dropout=0.2 --l2reg=0.0001 --seed=42']
# 调参小窍门
for cmd in cmds:
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    for line in iter(p.stdout.readline, b''):
        msg = line.strip().decode('gbk')
        print(msg)