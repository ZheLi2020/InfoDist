import torch
from fast_pytorch_kmeans import KMeans
from utils import get_network
import umap
import umap.plot


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def querybykmeans(images, num_classes, args):
    images = images.to(args.device)
    pre_model = get_network(args.embedding, 3, num_classes, (args.res, args.res), width=args.width,
                            depth=args.depth, args=args).to(args.device)
    model_path = ''
    if args.embedding == "ConvNet":
        model_path = f'model/{args.embedding}{args.res}depth{args.depth}.pth'
    elif args.embedding == "ResNet18":
        model_path = f'model/{args.embedding}{args.res}.pth'

    pre_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    embeddings = get_embeddings(images, pre_model, args.distributed)

    kmeans = KMeans(n_clusters=args.num_cluster, mode='euclidean')
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.centroids

    dist_matrix = euclidean_dist(centers, embeddings)
    min_indices = torch.topk(dist_matrix, k=args.subsample, largest=False, dim=1)[1]

    q_idxs = min_indices.view(-1)
    return q_idxs


def querybyumap(images, num_classes, args):
    images = images.to(args.device)
    pre_model = get_network(args.embedding, 3, num_classes, (args.res, args.res), width=args.width,
                            depth=args.depth, args=args).to(args.device)
    model_path = ''
    if args.depth == 5:
        model_path = f'model/{args.embedding}{args.res}.pth'
    elif args.depth == 6:
        model_path = f'model/{args.embedding}{args.res}depth6.pth'

    pre_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    embeddings = get_embeddings(images, pre_model, args.distributed)

    # display umap
    # for n in [5, 10, 15, 20, 25]:
    #     for m in [0.0, 0.02, 0.05, 0.1, 0.15]:
    #         mapper = umap.UMAP(n_neighbors=n, min_dist=m, random_state=42).fit(embeddings)
    #         p = umap.plot.points(mapper)
    #         umap.plot.show(p)

    kmeans = KMeans(n_clusters=args.num_cluster, mode='euclidean')
    mapper = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=args.reduced_dim).fit(embeddings)
    reduced_embedding = mapper.transform(embeddings)
    reduced_embedding = torch.from_numpy(reduced_embedding)
    labels = kmeans.fit_predict(reduced_embedding)
    centers = kmeans.centroids

    dist_matrix = euclidean_dist(centers, reduced_embedding)
    min_indices = torch.topk(dist_matrix, k=args.subsample, largest=False, dim=1)[1]

    q_idxs = min_indices.view(-1)
    return q_idxs


def querybyimageumap(images, args):
    images = images.to("cpu")
    kmeans = KMeans(n_clusters=args.num_cluster, mode='euclidean')
    images = images.view(images.size(0), -1)
    mapper = umap.UMAP(n_neighbors=10, min_dist=0.05, n_components=args.reduced_dim).fit(images)
    reduced_embedding = mapper.transform(images)
    reduced_embedding = torch.from_numpy(reduced_embedding)
    labels = kmeans.fit_predict(reduced_embedding)
    centers = kmeans.centroids

    dist_matrix = euclidean_dist(centers, reduced_embedding)
    min_indices = torch.topk(dist_matrix, k=args.subsample, largest=False, dim=1)[1]

    q_idxs = min_indices.view(-1)
    return q_idxs


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

