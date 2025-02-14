import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import argparse
import logging

from model import Model
import load_data
import metric

# torch.autograd.set_detect_anomaly(True)


def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='book_crossing')
    parser.add_argument("--social_data", type=bool, default=False)
    # test_set/cv/split
    parser.add_argument("--load_mode", type=str, default='test_set')

    parser.add_argument("--implcit_bottom", type=int, default=None, help="Minimum score threshold (for preprocessing)")
    # parser.add_argument("--cross_validate", type=int, default=None)
    # parser.add_argument("--split", type=float, default=None)
    parser.add_argument("--user_fre_threshold", type=int, default=None, help="Minimum user degree threshold (for preprocessing)")
    parser.add_argument("--item_fre_threshold", type=int, default=None, help="Minimum item degree threshold (for preprocessing)")

    parser.add_argument("--loadFilename", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs")

    parser.add_argument("--embedding_size", type=int, default=8, help="Item embedding size")
    parser.add_argument("--id_embedding_size", type=int, default=64, help="User/item ID embedding size")
    parser.add_argument("--dense_embedding_dim", type=int, default=16, help="Dense item embedding size")

    parser.add_argument("--L", type=int, default=3, help="Number of GCN layers")
    parser.add_argument("--link_topk", type=int, default=10, help="Number of similar items for item-item similarity matrix")

    parser.add_argument("--reg_lambda", type=float, default=0.02, help="Weight of popularity aware contrastive learning")
    parser.add_argument("--top_rate", type=float, default=0.1, help="Proportion of 'top' items")
    parser.add_argument("--convergence", type=float, default=40, help="Convergence rate of popularity coefficient")
    parser.add_argument("--seperate_rate", type=float, default=0.2, help="Ratio of head/tail items")
    parser.add_argument("--alpha", type=float, default=0.01, help="Learning rate of meta-training (alpha)")
    parser.add_argument("--beta", type=float, default=0.1, help="Weight of meta-training loss (beta)")

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate of downstream task training")
    # parser.add_argument("--weight_decay", type=float, default=0.01)  # unused

    parser.add_argument("--K_list", type=int, nargs='+', default=[10, 20, 50], help="Top-k list for metrics")
    parser.add_argument("--ssl_temp", type=float, default=1, help="Teperature in softmax")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID if available")

    opt = parser.parse_args()
    print(type(opt))

    return opt


def my_collate_train(batch):
    user_id = [item[0] for item in batch]
    pos_item = [item[1] for item in batch]
    neg_item = [item[2] for item in batch]

    user_id = torch.LongTensor(user_id)
    pos_item = torch.LongTensor(pos_item)
    neg_item = torch.LongTensor(neg_item)

    return [user_id, pos_item, neg_item]


def my_collate_test(batch):
    user_id = [item[0] for item in batch]
    test_item = [item[1] for item in batch]

    user_id = torch.LongTensor(user_id)
    test_item = torch.stack(test_item)

    # print("test batch shape:", user_id.shape, test_item.shape)
    return [user_id, test_item]


def collate_test_i2i(batch):
    item1 = [item[0] for item in batch]
    item2 = [item[1] for item in batch]

    item1 = torch.LongTensor(item1)
    item2 = torch.LongTensor(item2)

    return [item1, item2]


def create_matrix_from_idx(num_items, index_list):
    matrix = torch.zeros((len(index_list), num_items), dtype=torch.long)
    for i, index in enumerate(index_list):
        matrix[i, index] = 1
    return matrix.numpy()



def one_train(Data: load_data.Data, opt: argparse.Namespace):
    logger = logging.getLogger("MGL")
    logger.setLevel(logging.INFO)
    log_file = f"{opt.dataset_name}-epoch{opt.epoch}-batch{opt.batch_size}-alpha{opt.alpha}-beta{opt.beta}.log"
    f_handler = logging.FileHandler(log_file)
    f_formatter = logging.Formatter('%(levelname)s  %(asctime)s  [%(name)s] %(message)s')
    f_handler.setFormatter(f_formatter)
    f_handler.setLevel(logging.INFO)
    logger.addHandler(f_handler)

    s_handler = logging.StreamHandler()
    s_handler.setLevel(logging.INFO)
    s_formatter = logging.Formatter('%(message)s')
    logger.addHandler(s_handler)

    logger.info(str(opt))
    logger.info('Building dataloader >>>>>>>>>>>>>>>>>>>')

    test_dataset = Data.test_dataset
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=opt.batch_size, collate_fn=my_collate_test)

    device_id = opt.gpu
    device_name = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device_name}")
    device = torch.device(device_name)

    print(device)
    index = [Data.interact_train['userid'].tolist(), Data.interact_train['itemid'].tolist()]
    value = [1.0] * len(Data.interact_train)

    interact_matrix = torch.sparse_coo_tensor(index, value, (Data.user_num, Data.item_num)).to(device)

    i2i = torch.sparse.mm(interact_matrix.t(), interact_matrix)  # Item co-occurrance matrix (Equation 6)

    def sparse_where(A):
        A = A.coalesce()
        A_values = A.values()
        A_indices = A.indices()
        A_values = torch.where(A_values > 1, A_values.new_ones(A_values.shape), A_values)
        return torch.sparse_coo_tensor(A_indices, A_values, A.shape).to(A.device)

    i2i = sparse_where(i2i)  # Item co-occurrance matrix (Equation 6)

    def get_0_1_array(item_num, mask_rate=0.2):
        """Generate a random masking matrix on the item co-occurrence matrix.
        Args:
            item_num (int): Number of items
            rate (float, optional): Ratio of zero (masking) items. Defaults to 0.2.
        Returns:
            Torch sparse matrix
        """
        num_elems = item_num * item_num
        zeros_num = int(num_elems * mask_rate)
        new_array = np.ones(num_elems)
        new_array[:zeros_num] = 0
        np.random.shuffle(new_array)
        re_array = new_array.reshape(item_num, item_num)  # 2D-matrix
        re_array = torch.from_numpy(re_array).to_sparse().to(device)
        return re_array

    mask_rate = 1.0 - opt.top_rate
    mask = get_0_1_array(Data.item_num, mask_rate)
    i2i = mask * i2i * mask.t()  # Random masking of co-occurrance matrix (Equation 7)

    i2i = i2i.coalesce()

    item1 = i2i.indices()[0].tolist()
    item2 = i2i.indices()[1].tolist()
    i2i_pair = list(zip(item1, item2))


    logger.info("building model >>>>>>>>>>>>>>>")
    model = Model(Data, opt, device)


    logger.info('Building optimizers >>>>>>>')
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    logger.info('Start training...')
    start_epoch = 0
    # directory = directory_name_generate(model, opt, "no early stop")
    model = model.to(device)

    support_loader = DataLoader(i2i_pair, shuffle=True, batch_size=opt.batch_size, collate_fn=collate_test_i2i)  # Dataset of edge generator

    print_metrics = False

    for epoch in range(start_epoch, opt.epoch):
        model.train()

        train_loader = DataLoader(Data.train_dataset, shuffle=True, batch_size=opt.batch_size, collate_fn=my_collate_train)

        support_iter = iter(support_loader)

        with tqdm(total=len(train_loader), desc="epoch"+str(epoch)) as pbar:
            total_support_loss = 0.0
            total_query_loss = 0.0
            total_loss = 0.0

            for index, (user_id, pos_item, neg_item) in enumerate(train_loader):

                user_id = user_id.to(device)
                pos_item = pos_item.to(device)
                neg_item = neg_item.to(device)

                item1, item2 = next(support_iter)
                item1 = item1.to(device)
                item2 = item2.to(device)

                ## Meta-Train (Algorithm 1, Line 3)
                support_loss = model.i2i(item1, item2) + opt.reg_lambda * model.reg(item1)  # Compute loss function of the meta edge generator (Equation 15)

                ## Update parameter (Algorithm 1, Line 4)
                weight_for_local_update = list(model.generator.encoder.state_dict().values())
                grad = torch.autograd.grad(support_loss, model.generator.encoder.parameters(), create_graph=True, allow_unused=True)
                fast_weights = []
                for i, weight in enumerate(weight_for_local_update):
                    fast_weights.append(weight - opt.alpha * grad[i])

                ## Meta-test (Algorithm 1, Line 5)
                query_loss = model.q_forward(user_id, pos_item, neg_item, fast_weights)  # L_Rec

                ## Meta-optimization (Algorithm 1, Line 6)
                loss = query_loss + opt.beta * support_loss  # Final objective (Equation 17)

                total_support_loss += support_loss.item()
                total_query_loss += query_loss.item()
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)

        if print_metrics:
            model.eval()
            NDCG = defaultdict(list)
            RECALL = defaultdict(list)
            MRR = defaultdict(list)
            user_historical_mask = Data.user_historical_mask.to(device)
            with tqdm(total=len(train_loader), desc="epoch"+str(epoch)) as pbar:
                for i, (user_id, pos_item, _) in enumerate(train_loader):
                    user_id = user_id.to(device)
                    score = model.predict(user_id)
                    score = torch.mul(user_historical_mask[user_id], score).cpu().detach().numpy()
                    ground_truth = pos_item.detach().numpy()

                    num_items = score.shape[1]
                    gt_matrix = create_matrix_from_idx(num_items, ground_truth)

                    for K in opt.K_list:
                        ndcg, recall, mrr = metric.ranking_meansure_testset(score, gt_matrix, K, list(Data.trainSet_i.keys()))
                        NDCG[K].append(ndcg)
                        RECALL[K].append(recall)
                        MRR[K].append(mrr)

                    pbar.update(1)

            K = opt.K_list[0]
            logger.info("top-{}: NDCG: {:.5f}, RECALL: {:.5f}, MRR: {:.5f}".format(K, np.mean(NDCG[K]), np.mean(RECALL[K]), np.mean(MRR[K])))
        logger.info(f"Loss: meta_train: {total_support_loss:.5f}, meta_test: {total_query_loss:.5f}, overall: {total_loss:.5f}")

    last_checkpoint = {
        'sd': model.state_dict(),
        'opt':opt,
    }
    torch.save(last_checkpoint, 'model.tar')

# def one_test(Data, opt):
#     last_checkpoint = torch.load("model.tar")
    # remove the historical interaction in the prediction
    model = Model(Data, opt, device)
    # model.load_state_dict(best_checkpoint['sd'])
    model.load_state_dict(last_checkpoint['sd'])
    model = model.to(device)
    model.eval()
    user_historical_mask = Data.user_historical_mask.to(device)

    PRECISION = defaultdict(list)
    RECALL = defaultdict(list)
    MRR = defaultdict(list)
    NDCG = defaultdict(list)
    HR = defaultdict(list)

    head_NDCG = defaultdict(list)
    head_RECALL = defaultdict(list)
    tail_NDCG = defaultdict(list)
    tail_RECALL = defaultdict(list)
    body_NDCG = defaultdict(list)
    body_RECALL = defaultdict(list)

    with tqdm(total=len(test_loader), desc="predicting") as pbar:
        for i, (user_id, pos_item) in enumerate(test_loader):
            user_id = user_id.to(device)
            score = model.predict(user_id)
            score = torch.mul(user_historical_mask[user_id], score).cpu().detach().numpy()
            ground_truth = pos_item.detach().numpy()

            for K in opt.K_list:
                precision, recall, mrr, ndcg, hr = metric.ranking_meansure_testset(score, ground_truth, K, list(Data.testSet_i.keys()))
                head_ndcg, head_recall, tail_ndcg, tail_recall, body_ndcg, body_recall = metric.ranking_meansure_degree_testset(score, ground_truth, K, Data.itemDegrees, opt.seperate_rate, list(Data.testSet_i.keys()))

                PRECISION[K].append(precision)
                RECALL[K].append(recall)
                MRR[K].append(mrr)
                NDCG[K].append(ndcg)
                HR[K].append(hr)

                head_NDCG[K].append(head_ndcg)
                head_RECALL[K].append(head_recall)
                tail_NDCG[K].append(tail_ndcg)
                tail_RECALL[K].append(tail_recall)
                body_NDCG[K].append(body_ndcg)
                body_RECALL[K].append(body_recall)

            pbar.update(1)

    print(opt)
    logger.info(model.name)
    for K in opt.K_list:
        logger.info("---- Metrics@{} ----".format(K))
        logger.info("PRECISION@{}: {:.5f}".format(K, np.mean(PRECISION[K])))
        logger.info("RECALL@{}: {:.5f}".format(K, np.mean(RECALL[K])))
        logger.info("MRR@{}: {:.5f}".format(K, np.mean(MRR[K])))
        logger.info("NDCG@{}: {:.5f}".format(K, np.mean(NDCG[K])))
        logger.info("HR@{}: {:.5f}".format(K, np.mean(HR[K])))
        logger.info('\n')
        logger.info("head_NDCG@{}: {:.5f}".format(K, np.mean(head_NDCG[K])))
        logger.info("head_RECALL@{}: {:.5f}".format(K, np.mean(head_RECALL[K])))
        logger.info('\n')
        logger.info("tail_NDCG@{}: {:.5f}".format(K, np.mean(tail_NDCG[K])))
        logger.info("tail_RECALL@{}: {:.5f}".format(K, np.mean(tail_RECALL[K])))
        logger.info('\n')
        logger.info("body_NDCG@{}: {:.5f}".format(K, np.mean(body_NDCG[K])))
        logger.info("body_RECALL@{}: {:.5f}".format(K, np.mean(body_RECALL[K])))
        logger.info("\n")


opt: argparse.Namespace = get_config()
interact_train, interact_test, social, user_num, item_num, user_feature, item_feature = load_data.data_load(opt.dataset_name, social_data=opt.social_data, test_dataset= True, bottom=opt.implcit_bottom)
data = load_data.Data(interact_train, interact_test, social, user_num, item_num, user_feature, item_feature)
one_train(data, opt)
