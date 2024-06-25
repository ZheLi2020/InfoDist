import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from utils import get_dataset, get_network, get_eval_pool, ParamDiffAug, epoch
from eval_utils import get_eval_lrs
import random
from absl import logging
from ema_pytorch import EMA


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def eval_synset(it_run, net, testloader, args):
    net = net.to(args.device)

    ema = EMA(net, beta=0.995, power=1, update_after_step=0, update_every=1)
    criterion = nn.CrossEntropyLoss().to(args.device)

    start = time.time()

    with torch.no_grad():
        loss_test, acc_test, auc_test, fscore_test = epoch('test', testloader, ema, None, criterion, args, aug=False)

    time_test = time.time() - start
    logging.info(f'Evaluate_{it_run}: TestAcc:\t{acc_test}, AUC:\t{auc_test}, Fscore:\t{fscore_test}, eval time = {int(time_test)} s ')

    return net, acc_test, auc_test, fscore_test


def eval_loop(testloader=None, model_eval_pool=[], channel=3, num_classes=9, im_size=(32, 32), args=None):
    curr_acc_dict = {}
    curr_std_dict = {}
    eval_pool_dict = get_eval_lrs(args)
    max_epoch = {'ConvNet': 2400, 'ResNet18': 1500}

    for model_eval in model_eval_pool:
        start_model = time.time()
        logging.info(f'-------------------------\nTrain and Evaluation: model = {model_eval}')

        accs_test = []
        aucs_test = []
        fscores_test = []

        for it_run in range(args.num_run):
            net_model = get_network(model_eval, channel, num_classes, im_size, width=args.width,
                                   depth=args.depth, args=args).to(args.device)  # get a random model
            model_path = os.path.join(args.save_path, f'{model_eval}{max_epoch[model_eval]}_{it_run}.pth')
            logging.info(f'Load model from {model_path}')
            net_model.load_state_dict(torch.load(model_path))

            args.lr_net = eval_pool_dict[model_eval]
            trained_net, acc_score, auc_score, fscore = eval_synset(it_run, net_model, testloader, args=args)

            accs_test.append(acc_score)
            aucs_test.append(auc_score)
            fscores_test.append(fscore)

        logging.info(f'accs: {accs_test}, aucs: {aucs_test}, fscores: {fscores_test}')
        accs_test = np.array(accs_test)
        acc_test_mean = np.mean(accs_test)
        acc_test_std = np.std(accs_test)
        curr_acc_dict[model_eval] = acc_test_mean
        curr_std_dict[model_eval] = acc_test_std

        aucs_test = np.array(aucs_test)
        auc_test_mean = np.mean(aucs_test)
        fscores_test = np.array(fscores_test)
        fscores_test_mean = np.mean(fscores_test)

        logging.info(f'Evaluate Acc {len(accs_test)} random {model_eval}, '
                     f'mean = {acc_test_mean} std = {acc_test_std}\n-------------------------')
        logging.info(f'Evaluate AUC {len(aucs_test)} random {model_eval}, '
                     f'mean = {auc_test_mean} std = {np.std(aucs_test)}\n-------------------------')
        logging.info(f'Evaluate FScore {len(fscores_test)} random {model_eval}, '
                     f'mean = {fscores_test_mean} std = {np.std(fscores_test)}\n-------------------------')

        end_model = time.time()
        used_model = time.strftime("%H:%M:%S", time.gmtime(end_model - start_model))
        logging.info(f'The passed time of training model {model_eval} is {used_model}')

    logging.info({
        'Accuracy/Avg_All'.format(model_eval): np.mean(np.array(list(curr_acc_dict.values()))),
        'Std/Avg_All'.format(model_eval): np.mean(np.array(list(curr_std_dict.values())))})

    curr_acc_dict.pop("{}".format(args.model))
    curr_std_dict.pop("{}".format(args.model))

    logging.info({
        'Accuracy/Avg_Cross'.format(model_eval): np.mean(np.array(list(curr_acc_dict.values()))),
        'Std/Avg_Cross'.format(model_eval): np.mean(np.array(list(curr_std_dict.values())))})


def main(args):
    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    args.batch_train = 256
    args.batch_test = 128
    args.res = 64
    args.depth = 3
    # args.distributed = True

    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    run_dir = '20240225-163550-chocolate-universe-381'
    args.save_path = os.path.join(args.save_path, run_dir)

    start_time = time.time()
    set_logger(log_level='info', fname=os.path.join(args.save_path, 'output_eval_f1auc.log'))

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_train, args.res, args=args)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    logging.info('training begins')
    logging.info(f'Hyper-parameters: \n {args.__dict__}')
    logging.info(f'Evaluation model pool: {model_eval_pool}')

    eval_loop(testloader=testloader, model_eval_pool=model_eval_pool,
              channel=channel, num_classes=num_classes, im_size=im_size, args=args)

    end_time = time.time()
    used_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
    logging.info(f'The passed time is {used_time}')
    print('Training finished')


if __name__ == '__main__':
    import shared_args

    parser = shared_args.add_shared_args()
    args = parser.parse_args()

    main(args)


