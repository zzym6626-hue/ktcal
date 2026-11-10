import argparse
from .data import prepare_ais
from .experiment import train,train_sequence,evaluate
from .diagnostics import diagnose


def main():
    parser=argparse.ArgumentParser(description='KTCal paper software experiments')
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare',help='Prepare the labeled AIS2024 train partition')
    p.add_argument('--source',required=True); p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=42)
    p=commands.add_parser('train',help='Train with forward export or retain BiGRU for historical-window inference')
    p.add_argument('--manifest'); p.add_argument('--output'); p.add_argument('--resume')
    p.add_argument('--initialize-from',help='Start a separately recorded experiment from matching BiGRU weights')
    p.add_argument('--inference-mode',choices=['forward','bidirectional'])
    p.add_argument('--readout',choices=['last','final_hidden'])
    p.add_argument('--scheduler',choices=['none','cosine'])
    p.add_argument('--stop-after-epochs',type=int,help='Save and return after this many epochs; resume with last.pt')
    p.add_argument('--variant',choices=['full','no_temporal','no_gru','full_conv'])
    for key in ('epochs','patience','batch-size','stride','val-stride','seed','threads'):
        p.add_argument('--'+key,type=int)
    for key in ('lr','min-lr','finetune-lr','weight-decay','forward-weight'):
        p.add_argument('--'+key,type=float)
    p.add_argument('--device'); p.add_argument('--purpose')
    p=commands.add_parser('train-seq',
        help='Train the full BiGRU on record chunks with every timestep supervised')
    p.add_argument('--manifest'); p.add_argument('--output'); p.add_argument('--resume')
    p.add_argument('--initialize-from',
        help='PaperSTNet checkpoint or models.STNet sequence checkpoint for warm start')
    p.add_argument('--stop-after-epochs',type=int,
        help='Save and return after this many epochs; resume with last.pt')
    p.add_argument('--scheduler',choices=['none','cosine'])
    p.add_argument('--readout',choices=['last','final_hidden'])
    for key in ('epochs','patience','chunk-window','batch-size','val-stride','seed','threads'):
        p.add_argument('--'+key,type=int)
    for key in ('lr','min-lr','weight-decay','match-weight'):
        p.add_argument('--'+key,type=float)
    p.add_argument('--device'); p.add_argument('--purpose')
    p=commands.add_parser('evaluate',help='Score immutable online predictions; save tables and diagnostics')
    p.add_argument('--checkpoint',required=True); p.add_argument('--manifest',required=True); p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['val','holdout'],default='holdout')
    p.add_argument('--mode',choices=['stnet_only','ktcal','both'],default='both')
    p.add_argument('--device',default='auto'); p.add_argument('--threads',type=int,default=4)
    p.add_argument('--tau1',type=float,default=3.); p.add_argument('--tau2',type=float,default=5.)
    p.add_argument('--sigma',type=float,default=1.); p.add_argument('--realtime',action='store_true')
    p.add_argument('--q-pos',type=float,default=.1); p.add_argument('--q-vel',type=float,default=.5)
    p.add_argument('--r-px',type=float,default=2.)
    p.add_argument('--capture',choices=['fine','coarse'],default='fine',
        help='coarse adds a factor-2 proposal pass with about +-8 px capture')
    p.add_argument('--min-psr',type=float,default=None,
        help='Reject matches whose peak-to-sidelobe ratio falls below this value')
    p.add_argument('--border-tier',type=int,choices=[2,3],default=3,
        help='Tier used when the NCC peak touches the response border')
    p.add_argument('--ignore-border-peak',action='store_true',
        help='Disable border-peak escalation (closer to the literal paper gate)')
    p.add_argument('--fusion-alpha',type=float,default=None,
        help='Blend alpha*STNet + (1-alpha)*Kalman on every frame (accuracy-first, always calls STNet)')
    p.add_argument('--allow-partial',action='store_true',
        help='Evaluate a best-so-far bidirectional checkpoint that was never finalized')
    p.add_argument('--subpixel',action='store_true',
        help='Refine the NCC peak with a parabola fit instead of the 1 px argmax lattice')
    p.add_argument('--max-frames',type=int,help='Pilot only: limit prefix length of each recording')
    p=commands.add_parser('diagnose',help='Compare train/val predictions and original states without opening holdout')
    p.add_argument('--checkpoint',required=True); p.add_argument('--manifest',required=True); p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['train','val'],default='val')
    p.add_argument('--stride',type=int,default=16); p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--device',default='auto'); p.add_argument('--threads',type=int,default=4)
    p.add_argument('--plot',action='store_true',help='Save event/label/prediction overlays; requires matplotlib')
    args=vars(parser.parse_args()); command=args.pop('command')
    if command=='prepare': result=prepare_ais(**args)
    elif command=='train':
        args['manifest_path']=args.pop('manifest'); result=train(**args)
    elif command=='train-seq':
        args['manifest_path']=args.pop('manifest'); result=train_sequence(**args)
    else:
        args['manifest_path']=args.pop('manifest')
        if args.pop('ignore_border_peak',False): args['border_tier']=None
        result=diagnose(**args) if command=='diagnose' else evaluate(**args)
    print(result,flush=True)


if __name__=='__main__': main()
