import sys
import argparse
import logging

from utils.trainer import train_model
from utils.validator import val_model


def main(args):
    if args.mode == 'train':
        flag = train_model(args)
        if not flag:
            logging.error("Training failed due to config issues.")
            sys.exit(1)
    elif args.mode == 'val':
        val_model(args)



if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='the config file to use')
    parser.add_argument('--mode', choices=['train', 'val'], required=True)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        '--predictions-out',
        default='',
        help='optional JSON path for exporting post-processed per-video predictions in validation mode',
    )
    parser.add_argument(
        '--map-eval-scope',
        choices=['global', 'video'],
        default='global',
        help='mAP evaluation scope for --mode val: global dataset AP or per-video averaged AP',
    )

    args = parser.parse_args()

    main(args)
