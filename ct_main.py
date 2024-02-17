import argparse
import torch
import yaml
from CT.ct import CT_enc, CT_enc_gcn, CT_enc_tnt, CT_enc_one, CT_enc_d2

parser = argparse.ArgumentParser(description='[CT] contrastive traffic prediction')
parser.add_argument('--model', type=str, default='staeformer', help='model name')
parser.add_argument('--road', type=str, default='er', help='road name')
parser.add_argument('--dataset', type=str, default='ct', help='in or out ')
parser.add_argument('--i_o', type=str, default='in', help='in or out ')
parser.add_argument('--do_pred', type=bool, default=False, help='predict or not')
parser.add_argument('--enc_dec', type=bool, default=False, help='encoder-decoder or not')

with open(f'{parser.parse_args().dataset}.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg = cfg[parser.parse_args().model]
parser.add_argument('--cfg', type=dict, default=cfg, help='model config')
parser.add_argument('--itr', type=int, default=1, help='iteration')

args = parser.parse_args()
args.road = {'er': '二环路', 'san': '三环路', 'si': '四环路', 'wu': '五环路', 'liu': '六环路', 'all': 'all'}[args.road]
cfg["use_gpu"] = True if torch.cuda.is_available() and cfg["use_gpu"] else False
CT = {'tnt': CT_enc_tnt,
      'staeformer': CT_enc,
      'itransformer': CT_enc,
      'patchtst': CT_enc_one,
      'stj': CT_enc,
      'stj_': CT_enc,
      'gat': CT_enc_gcn,
      'gat2': CT_enc_gcn,
      'std': CT_enc_gcn,
      'd2stgnn': CT_enc_d2}[args.model]
for ii in range(args.itr):
    setting = '{}_{}_{}_{}'.format(args.i_o, args.model, cfg["date_range"], ii)
    print('>>>>>>>.....start training.....>>>>>>>>>>>>>>>>>>>>>>>>>>')
    ct = CT(args)
    ct.build_log(setting)
    print('>>>>>>>.....training..... <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    if not args.do_pred:
        ct.train(setting, True)

    print('>>>>>>>.....testing......<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    ct.test(setting, args.do_pred)

    ct.log.close()





