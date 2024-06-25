import numpy as np
import torch
from fast_pytorch_kmeans import KMeans
from utils import get_network
import umap
import umap.plot
from infomap import Infomap
from absl import logging
import torch.nn.functional as F


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def construct_graph(im, graph):
    batch_size = graph.shape[0]
    im.add_nodes(range(batch_size))
    weights = []
    for n in range(batch_size):
        idx = np.where(graph[n] > 0)[0]
        for l in idx:
            weight = graph[n][l]
            im.add_link(n, l, weight=weight)

        wei = graph[n][idx]
        weights.append(np.sum(wei))

    return im, weights


def construct_graph_norm(im, graph, th):
    batch_size = graph.size(0)
    im.add_nodes(range(batch_size))
    weights = []
    for n in range(batch_size):
        n_weights = 1 / graph[n]
        n_weights[n] = 0
        n_weights = F.softmax(n_weights).numpy()
        n_weights[n_weights < th] = 0
        idx = np.where(n_weights > 0)[0]

        # n_weights = n_weights/np.max(n_weights)
        for l in idx:
            im.add_link(n, l, weight=n_weights[l])

        weights.append(np.sum(n_weights))

    return im, weights


def construct_graph_median(im, graph):
    batch_size = graph.size(0)
    im.add_nodes(range(batch_size))
    weights = []
    for n in range(batch_size):
        n_weights = 1 / graph[n]
        n_weights[n] = 0
        n_weights = F.softmax(n_weights).numpy()
        th = np.median(n_weights)
        n_weights[n_weights < th] = 0
        idx = np.where(n_weights > 0)[0]

        # n_weights = n_weights/np.max(n_weights)
        for l in idx:
            im.add_link(n, l, weight=n_weights[l])

        weights.append(np.sum(n_weights))

    return im, weights


def construct_graph_infonorm(im, graph):
    batch_size = graph.size(0)
    im.add_nodes(range(batch_size))
    weights = []
    for n in range(batch_size):
        n_weights = graph[n]
        idx = np.where(n_weights > 0)[0]
        n_weights = 1/n_weights[idx]
        n_weights = F.softmax(n_weights).numpy()

        # n_weights = n_weights/np.max(n_weights)
        count = 0
        for l in idx:
            im.add_link(n, l, weight=n_weights[count])
            count += 1

        weights.append(np.sum(n_weights))

    return im, weights


def querybyumapinfouniform_high(images, args):
    images = images.to("cpu")
    num_imgs = images.size(0)
    images = images.view(num_imgs, -1)
    # correlation metrics: cosine, correlation, euclidean
    mapper = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=0.05, n_components=args.reduced_dim, metric=args.umapmetric).fit(images)

    im = Infomap("--two-level --directed")

    if args.norm:
        dist_matrix = torch.from_numpy(mapper.graph_.A)
        im, weights = construct_graph_infonorm(im, dist_matrix)
    else:
        im, weights = construct_graph(im, mapper.graph_.A)
    im.run("--num-trials 5")
    logging.info(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")

    idx = 0
    inx2id = {}
    indices_modules = [[] for c in range(im.num_top_modules)]
    for node in im.iterLeafNodes():
        # print(f"node id {node.node_id}, weight id {node.module_id}")
        # node.data.flow, weights[node.node_id], node.data.exit_flow, node.data.enter_flow, node.modular_centrality
        if args.rankmetric == 'modular_centrality':
            indices_modules[node.module_id - 1].append([node.node_id, node.modular_centrality])
        elif args.rankmetric == 'enter_flow':
            indices_modules[node.module_id - 1].append([node.node_id, node.data.enter_flow])
        elif args.rankmetric == 'exit_flow':
            indices_modules[node.module_id - 1].append([node.node_id, node.data.exit_flow])
        else:
            indices_modules[node.module_id - 1].append([node.node_id, weights[node.node_id]])
        inx2id[node.node_id] = node.module_id
        idx += 0

    num_subimgs = 100
    selected_idx = []
    avg_num = int(num_subimgs / im.num_top_modules)
    rest_num = int(num_subimgs % im.num_top_modules)
    for i, m in enumerate(indices_modules):
        if i < rest_num:
            sliced_pair = m[:avg_num + 1]
            selected_idx += [s[0] for s in sliced_pair]
        else:
            if avg_num > 0:
                sliced_pair = m[:avg_num]
                selected_idx += [s[0] for s in sliced_pair]

    if len(selected_idx) < num_subimgs:
        logging.info(f'uniform select {num_subimgs} images, less than target, the number of selected images is {len(selected_idx)}')
        num_rest = num_subimgs - len(selected_idx)
        sliced_pair = indices_modules[0][avg_num+1: avg_num+num_rest+1]
        temp_idx = [s[0] for s in sliced_pair]
        selected_idx += temp_idx

    return selected_idx


def querybyembedinfo_high(images, num_classes, args):
    images = images.to(args.device)
    num_imgs = images.size(0)

    pre_model = get_network(args.embedding, 3, num_classes, (args.res, args.res), width=args.width,
                            depth=args.depth, args=args).to(args.device)
    model_path = ''
    if args.embedding == "ConvNet":
        model_path = f'model/{args.embedding}{args.res}depth{args.depth}.pth'
    elif args.embedding == "ResNet18":
        model_path = f'model/{args.embedding}{args.res}.pth'

    pre_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    embeddings = get_embeddings(images, pre_model, args.distributed)
    dist_matrix = euclidean_dist(embeddings, embeddings)
    # dist_matrix = F.softmax(dist_matrix, dim=1)

    im = Infomap("--two-level --directed")
    if args.th_auto == 'norm':
        im, weights = construct_graph_norm(im, dist_matrix, args.threshold)
    elif args.th_auto == 'median':
        im, weights = construct_graph_median(im, dist_matrix)
    im.run("--num-trials 5")
    logging.info(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")

    idx = 0
    inx2id = {}
    indices_modules = [[] for c in range(im.num_top_modules)]
    for node in im.iterLeafNodes():
        # print(f"node id {node.node_id}, weight id {node.module_id}")
        # node.data.flow, weights[node.node_id], node.data.exit_flow, node.data.enter_flow, node.modular_centrality
        if args.rankmetric == 'modular_centrality':
            indices_modules[node.module_id - 1].append([node.node_id, node.modular_centrality])
        elif args.rankmetric == 'enter_flow':
            indices_modules[node.module_id - 1].append([node.node_id, node.data.enter_flow])
        elif args.rankmetric == 'exit_flow':
            indices_modules[node.module_id - 1].append([node.node_id, node.data.exit_flow])
        else:
            indices_modules[node.module_id - 1].append([node.node_id, weights[node.node_id]])
        inx2id[node.node_id] = node.module_id
        idx += 0

    num_subimgs = 100
    selected_idx = []

    # select 10 percent 100 images
    avg_num = int(num_subimgs / im.num_top_modules)
    rest_num = int(num_subimgs % im.num_top_modules)
    for i, m in enumerate(indices_modules):
        if i < rest_num:
            sliced_pair = m[:avg_num + 1]
            selected_idx += [s[0] for s in sliced_pair]
        else:
            if avg_num > 0:
                sliced_pair = m[:avg_num]
                selected_idx += [s[0] for s in sliced_pair]

    if len(selected_idx) < num_subimgs:
        logging.info(f'less than 10 percent, the number of selected images is {len(selected_idx)}')
        num_rest = num_subimgs - len(selected_idx)
        sliced_pair = indices_modules[0][avg_num+1: avg_num+num_rest+1]
        temp_idx = [s[0] for s in sliced_pair]
        selected_idx += temp_idx

    return selected_idx


def get_embeddings(images, model, distributed):
    if distributed:
        embed = model.module.embed
    else:
        embed = model.embed
    features = []
    num_img = images.size(0)
    batch_size = 64
    with torch.no_grad():
        for i in range(0, num_img, batch_size):
            subimgs = images[i:i+batch_size]
            subfeatures = embed(subimgs).detach()
            features.append(subfeatures)

    features = torch.cat(features, dim=0).to("cpu")
    return features


def get_probabilities(images, model):
    out = model(images)
    return out

