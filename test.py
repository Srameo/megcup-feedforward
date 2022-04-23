import os
import argparse
from tqdm import tqdm
import megengine as mge

from feedback_restormer import FeedbackRestormer
from dataset import make_dataloader
from utils import pixel_shuffle

def make_model(ckpt):
    model = FeedbackRestormer()
    model.load_state_dict(mge.load(ckpt), strict=True)
    return model

def parse_args():
    parser = argparse.ArgumentParser('MegCup 2022 FeedForward', add_help=False)
    parser.add_argument('--data-path', required=True, type=str, help="test data path")
    parser.add_argument('--batch-size', type=int, default=1, help="batch size")
    parser.add_argument('--num-workers', type=int, default=0, help="num_workers of dataloader")
    parser.add_argument('--checkpoint', required=True, type=str, help='checkpoint path')
    parser.add_argument('--output', type=str, default='.', metavar='PATH', help='where to make the bin')

    args, _ = parser.parse_known_args()
    return args

def test(args):
    print('Test Begin!')
    print('Make Loader...')
    loader = make_dataloader(args.data_path, args.batch_size, args.num_workers)
    print('Make Model...')
    model = make_model(args.checkpoint)
    print(f'Params: {sum(p.size for p in model.parameters()) / 1000.0}k')
    basename = os.path.basename(args.checkpoint)
    fout = open(os.path.join(f"{args.output}", f'{basename}_prediction.0.2.bin'), 'wb')
    model.eval()
    print('Testing ...')
    for x in tqdm(loader):
        x = mge.tensor(x, 'float32')
        y = model(x)
        yout = pixel_shuffle(y.numpy().copy() * 65535).clip(0, 65535).astype('uint16')
        fout.write(yout.tobytes())

    fout.close()
    print('Test Finished!')

if __name__ == '__main__':
    test(parse_args())
