import argparse
from .data import prepare_ais
from .experiment import train,evaluate
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
    p=commands.add_parser('evaluate',help='Score immutable online predictions; save tables and diagnostics')
    p.add_argument('--checkpoint',required=True); p.add_argument('--manifest',required=True); p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['val','holdout'],default='holdout')
    p.add_argument('--mode',choices=['stnet_only','ktcal','both'],default='both')
    p.add_argument('--device',default='auto'); p.add_argument('--threads',type=int,default=4)
    p.add_argument('--tau1',type=float,default=3.); p.add_argument('--tau2',type=float,default=5.)
    p.add_argument('--sigma',type=float,default=1.); p.add_argument('--realtime',action='store_true')
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
    else:
        args['manifest_path']=args.pop('manifest')
        result=diagnose(**args) if command=='diagnose' else evaluate(**args)
    print(result,flush=True)


if __name__=='__main__': main()
