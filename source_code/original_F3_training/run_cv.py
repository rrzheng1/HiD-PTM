import argparse
import subprocess
import sys
from common import HERE,VARIANTS

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda')
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--patience',type=int,default=5);parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--min-delta',type=float,default=.0001)
    parser.add_argument('--variant',choices=VARIANTS,default='F0');args=parser.parse_args()
    for fold in range(1,6):
        subprocess.run([sys.executable,str(HERE/'train.py'),'--fold',str(fold),'--device',args.device,'--seed',str(args.seed),
                        '--epochs',str(args.epochs),'--patience',str(args.patience),'--batch-size',str(args.batch_size),'--variant',args.variant,
                        '--min-delta',str(args.min_delta)],check=True)
    subprocess.run([sys.executable,str(HERE/'summarize.py'),'--seed',str(args.seed),'--variant',args.variant],check=True)

if __name__=='__main__':main()
