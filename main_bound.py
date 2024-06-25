import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from utils_bound import get_dataset, get_network, get_eval_pool, ParamDiffAug, train_synset
from eval_utils import build_dataset, choose_dataset, choose_random, get_eval_lrs
import random
import wandb
from absl import logging


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


def train_loop(images_all=None, labels_all=None, indices_class=None, testloader=None,
              model_eval_pool=[], channel=3, num_classes=9, im_size=(64, 64), args=None):
    curr_acc_dict = {}
    curr_std_dict = {}
    eval_pool_dict = get_eval_lrs(args)

    for model_eval in model_eval_pool:
        start_model = time.time()
        logging.info(f'-------------------------\nTrain and Evaluation: model = {model_eval}')

        accs_test = []
        aucs_test = []
        fscores_test = []

        for it_run in range(args.num_run):
            image_small, label_small, indices_class_small = choose_dataset(images_all, labels_all, indices_class, num_classes, args)
            net_model = get_network(model_eval, channel, num_classes, im_size, width=args.width,
                                   depth=args.depth, args=args).to(args.device)  # get a random model

            args.lr_net = eval_pool_dict[model_eval]
            trained_net, acc_test, auc_test, fscore_test = train_synset(it_run, net_model, image_small, label_small,
                                                                        testloader, args=args, aug=True)
            torch.save(trained_net.state_dict(), os.path.join(args.save_path, f'{model_eval}_{it_run}.pth'))
            accs_test.append(acc_test)
            aucs_test.append(auc_test)
            fscores_test.append(fscore_test)

        logging.info(f'accs: {accs_test}')
        logging.info(f'aucs: {aucs_test}')
        logging.info(f'fscores: {fscores_test}')
        accs_test = np.array(accs_test)
        acc_test_mean = np.mean(accs_test)
        acc_test_std = np.std(accs_test)
        best_dict_str = "{}".format(model_eval)

        curr_acc_dict[best_dict_str] = acc_test_mean
        curr_std_dict[best_dict_str] = acc_test_std

        aucs_test = np.array(aucs_test)
        auc_test_mean = np.mean(aucs_test)
        fscores_test = np.array(fscores_test)
        fscores_test_mean = np.mean(fscores_test)

        logging.info(f'Evaluate ACC {len(accs_test)} random {model_eval}, '
                     f'mean = {acc_test_mean} std = {acc_test_std}\n----------------------')
        logging.info(f'Evaluate AUC {len(aucs_test)} random {model_eval}, '
                     f'mean = {auc_test_mean} std = {np.std(aucs_test)}\n----------------------')
        logging.info(f'Evaluate FScore {len(fscores_test)} random {model_eval}, '
                     f'mean = {fscores_test_mean} std = {np.std(fscores_test)}\n--------------------')

        wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean})
        wandb.log({'Std/{}'.format(model_eval): acc_test_std})
        wandb.log({'AUC_Score/{}'.format(model_eval): auc_test_mean})
        wandb.log({'FScore/{}'.format(model_eval): fscores_test_mean})

        end_model = time.time()
        used_model = time.strftime("%H:%M:%S", time.gmtime(end_model - start_model))
        logging.info(f'The passed time of training model {model_eval} is {used_model}')

    wandb.log({
        'Accuracy/Avg_All'.format(model_eval): np.mean(np.array(list(curr_acc_dict.values()))),
        'Std/Avg_All'.format(model_eval): np.mean(np.array(list(curr_std_dict.values())))})

    curr_acc_dict.pop("{}".format(args.model))
    curr_std_dict.pop("{}".format(args.model))

    wandb.log({
        'Accuracy/Avg_Cross'.format(model_eval): np.mean(np.array(list(curr_acc_dict.values()))),
        'Std/Avg_Cross'.format(model_eval): np.mean(np.array(list(curr_std_dict.values())))})


def main(args):
    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # print('Start with sleep')
    # time.sleep(8000)  # delay 4000 seconds

    args.contrastive = True

    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    run = wandb.init(project="medmnist", job_type="synthetic", config=args, )
    run_dir = "{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), run.name)
    args.save_path = os.path.join(args.save_path, run_dir)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    start_time = time.time()
    set_logger(log_level='info', fname=os.path.join(args.save_path, 'output.log'))

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_train, args.res, args=args)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    # args.distributed = torch.cuda.device_count() > 1
    images_all, labels_all, indices_class = build_dataset(dst_train, class_map, num_classes)

    logging.info('training begins')
    logging.info(f'Hyper-parameters: \n {args.__dict__}')
    logging.info(f'Evaluation model pool: {model_eval_pool}')

    train_loop(images_all=images_all, labels_all=labels_all, indices_class=indices_class,
               testloader=testloader, model_eval_pool=model_eval_pool, channel=channel,
              num_classes=num_classes, im_size=im_size, args=args)

    end_time = time.time()
    used_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
    logging.info(f'The passed time is {used_time}')
    print('Training finished')


if __name__ == '__main__':
    import shared_args

    parser = shared_args.add_shared_args()
    args = parser.parse_args()

    main(args)


