import os
import logging
import numpy as np

def pad_collate_seq(seq_inp_batch):
    # seq_inp: list of unit_seq
    # unit_seq: list of nb_floors * 5
    data_length = []
    X_seq_new = []
    for unit_seq in seq_inp_batch:
        unit_seq.sort(key=lambda x: len(x), reverse=True)
        data_length.append([len(u) for u in unit_seq])
        unit_padded = pad_sequence(list(unit_seq), batch_first=True, padding_value=0)
        X_seq_new.append(unit_padded)
    return X_seq_new, data_length

class GOATLogger:

    def __init__(self, args):
        self.mode = args.mode
        self.save_root = args.save_path

        if self.mode == 'train':
            if not os.path.exists(self.save_root):
                os.mkdir(self.save_root)
            filename = os.path.join(self.save_root, 'console.log')
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s.%(msecs)03d - %(message)s',
                                datefmt='%b-%d %H:%M:%S',
                                filename=filename,
                                filemode='w')
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(logging.Formatter('%(message)s'))
            logging.getLogger('').addHandler(console)

            logging.info("Logger created at {}".format(filename))
        else:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s.%(msecs)03d - %(message)s',
                                datefmt='%b-%d %H:%M:%S')

        logging.info("Random Seed: {}".format(args.seed))
        logging.info(args)
        self.reset_stats()

    def reset_stats(self):
        if self.mode == 'train':
            self.stats = {'train': {'loss': [], 'mae': [], 'rmse': []},
                          'eval': {'loss': [], 'mae': [], 'rmse': []}}
        else:
            self.stats = {'eval': {'loss': [], 'mae': [], 'rmse': []}}



    def logdebug(self, strout):
        logging.debug(strout)

    def loginfo(self, strout):
        logging.info(strout)