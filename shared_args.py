import argparse
import json

def add_shared_args():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='medsyn', help='dataset')
    parser.add_argument('--device', type=str, default='cuda:2', help='GPU ID')
    parser.add_argument('--device_ids', '--list', type=json.loads, default=[0, 1, 2, 3], help='used in multiple GPUs')
    parser.add_argument('--model', type=str, default='ConvNet', help='distillation model')
    parser.add_argument('--embedding', type=str, default='ConvNet', help='embedding models')
    parser.add_argument('--cluster', type=str, default='embedinfo', help='cluster methods, kmeans or infomap')
    parser.add_argument('--reduced_dim', type=int, default=10, help='used for UMAP to reduce feature dimension')
    parser.add_argument('--num_cluster', type=int, default=100, help='number of clusters')
    parser.add_argument('--subsample', type=int, default=1, help='number of samples from each cluster')
    parser.add_argument('--alpha', type=float, default=0.1, help='hyperparameter for contrastive loss')
    parser.add_argument('--n_neighbors', type=int, default=10, help='number of neighbors in UMAP')
    parser.add_argument('--umapmetric', type=str, default='euclidean', help='distance metric in UMAP')
    parser.add_argument('--rankmetric', type=str, default='modular_centrality', help='metric of representative samples')
    parser.add_argument('--threshold', type=float, default=0.001, help='threshold to construct graph for infomap.')
    parser.add_argument('--th_auto', type=str, default='norm', help='method to construct graph for infomap, can be median')

    parser.add_argument('--eval_mode', type=str, default='M', help='eval_mode')  # M: multi architectures
    parser.add_argument('--num_run', type=int, default=5, help='the number of evaluating randomly initialized models')
    parser.add_argument('--total_epoch', type=int, default=1000,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--batch_train', type=int, default=64, help='batch size for training networks')
    parser.add_argument('--batch_test', type=int, default=64, help='batch size for testing networks')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='../U-ViT/output/medmnist2024-03-06_08-16-56/eval_samples_20000', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')

    parser.add_argument('--res', type=int, default=64, choices=[64, 256], help='resolution')
    parser.add_argument('--norm', action='store_true', help='to apply normalization for UMAP graph')
    parser.add_argument('--distributed', action='store_true', help='to use multiple GPUs')
    parser.add_argument('--contrastive', action='store_true', help='to add contrastive loss')

    parser.add_argument('--width', type=int, default=128, help='width of ConvNet')
    parser.add_argument('--depth', type=int, default=5, help='depth of ConvNet')

    parser.add_argument('--pos_beta', default=0.9, type=float, metavar='L',
                        help='position hyperparameter for boundary in contrastive loss')
    parser.add_argument('--margin_tau', default=0.3, type=float, metavar='L',
                        help='margin hyperparameter between positive and negative boundary in contrastive loss')

    return parser
