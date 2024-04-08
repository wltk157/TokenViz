import os.path
import pandas
import torch
import sys

sys.path.append("..")

from shared.visualize.data_loader import load_data
from shared.visualize.word2vec import train_vord2vec_and_save, word2vec_net
from shared.visualize.code2block import tokenize_code, TokenVisDataset


def block_visualize(dataset_file, IR_path, row, col, blk_height, blk_width, variable, newline):
    original_dataset = pandas.read_csv(dataset_file)

    print("Loading pretrain data...")
    codes = [tokenize_code(code) for code in original_dataset['code'].tolist()]
    batch_size, max_window_size, num_noise_words = 512, 5, 5
    data_iter, vocab = load_data(codes, batch_size, max_window_size,
                                 num_noise_words, 5)
    lr, num_epochs, embed_size = 0.002, 100, blk_width * blk_height

    net = word2vec_net(len(vocab), embed_size)

    if not os.path.isfile('word2vec_block.pth'):
        print('Pretrain model not found, start training...')
        train_vord2vec_and_save(net, data_iter, lr, num_epochs, save_path='word2vec_block.pth')
    else:
        print('Pretrain model founded, loading state dict...')
        net.load_state_dict(torch.load('word2vec_block.pth'))

    dataset = TokenVisDataset(net[0], vocab, row, col, blk_width, blk_height, variable, newline, folder=IR_path)
    dataset.build(zip(original_dataset['id'].tolist(), original_dataset['code'].tolist()))
