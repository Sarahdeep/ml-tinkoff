import argparse
import pickle

from train import NgramModel


def parse_cmd_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', '--len', type=int, nargs='?', default=30)
    parser.add_argument('--model', type=argparse.FileType('rb'), nargs='?', default='./model.pkl')
    parser.add_argument('--prefix', type=str, nargs='*')
    return parser.parse_args()


if __name__ == "__main__":
    model: NgramModel
    args = parse_cmd_line_arguments()
    model = pickle.load(args.model)
    print(model.generate_text(args.length, args.prefix))
