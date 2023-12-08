import numpy as np
import bottleneck as bn
import torch
import random
from tqdm import tqdm
from rectorch.metrics import Metrics


def precision_at_k(pred_scores, ground_truth, k=100):
    assert pred_scores.shape == ground_truth.shape, "'pred_scores' and 'ground_truth' must have the same shape."
    k = min(pred_scores.shape[1], k)
    idx = bn.argpartition(-pred_scores, k-1, axis=1)
    pred_scores_binary = np.zeros_like(pred_scores, dtype=bool)
    pred_scores_binary[np.arange(pred_scores.shape[0])[:, np.newaxis], idx[:, :k]] = True
    X_true_binary = (ground_truth > 0)
    num = (np.logical_and(X_true_binary, pred_scores_binary).sum(axis=1)).astype(np.float32)
    precision = num / k
    return precision


def ranking_meansure_testset(pred_scores, ground_truth, k, test_item):
    pred_scores = pred_scores[:, test_item]
    ground_truth = ground_truth[:, test_item]

    # user_num
    precision_list = precision_at_k(pred_scores, ground_truth, k).tolist()
    recall_list = Metrics.recall_at_k(pred_scores, ground_truth, k).tolist()
    mrr_list = Metrics.mrr_at_k(pred_scores, ground_truth, k).tolist()
    ndcg_list = Metrics.ndcg_at_k(pred_scores, ground_truth, k).tolist()
    hr_list = Metrics.hit_at_k(pred_scores, ground_truth, k).tolist()

    return np.mean(precision_list), np.mean(recall_list), np.mean(mrr_list), np.mean(ndcg_list), np.mean(hr_list)


def ranking_meansure_degree_testset(pred_scores, ground_truth, k, itemDegrees, seperate_rate, test_item):
    sorted_item_degrees = sorted(itemDegrees.items(), key=lambda x: x[1])
    item_list_sorted, _ = zip(*sorted_item_degrees)
    body_length = int(len(item_list_sorted) * (1-seperate_rate))
    tail_length = int(len(item_list_sorted) * seperate_rate)
    head_length = int(len(item_list_sorted) * seperate_rate)

    head_item = list(set(item_list_sorted[-head_length:]).intersection(set(test_item)))
    tail_item = list(set(item_list_sorted[:tail_length]).intersection(set(test_item)))
    body_item = list(set(item_list_sorted[:body_length]).intersection(set(test_item)))

    head_precision_list = np.nan_to_num(precision_at_k(pred_scores[:, head_item], ground_truth[:, head_item], k)).tolist()
    head_recall_list = np.nan_to_num(Metrics.recall_at_k(pred_scores[:, head_item], ground_truth[:, head_item], k)).tolist()
    head_mrr_list = np.nan_to_num(Metrics.mrr_at_k(pred_scores[:, head_item], ground_truth[:, head_item], k)).tolist()
    head_ndcg_list = np.nan_to_num(Metrics.ndcg_at_k(pred_scores[:, head_item], ground_truth[:, head_item], k)).tolist()
    head_hr_list = np.nan_to_num(Metrics.hit_at_k(pred_scores[:, head_item], ground_truth[:, head_item], k)).tolist()

    tail_precision_list = np.nan_to_num(precision_at_k(pred_scores[:, tail_item], ground_truth[:, tail_item], k)).tolist()
    tail_recall_list = np.nan_to_num(Metrics.recall_at_k(pred_scores[:, tail_item], ground_truth[:, tail_item], k)).tolist()
    tail_mrr_list = np.nan_to_num(Metrics.mrr_at_k(pred_scores[:, tail_item], ground_truth[:, tail_item], k)).tolist()
    tail_ndcg_list = np.nan_to_num(Metrics.ndcg_at_k(pred_scores[:, tail_item], ground_truth[:, tail_item], k)).tolist()
    tail_hr_list = np.nan_to_num(Metrics.hit_at_k(pred_scores[:, tail_item], ground_truth[:, tail_item], k)).tolist()

    body_precision_list = np.nan_to_num(precision_at_k(pred_scores[:, body_item], ground_truth[:, body_item], k)).tolist()
    body_recall_list = np.nan_to_num(Metrics.recall_at_k(pred_scores[:, body_item], ground_truth[:, body_item], k)).tolist()
    body_mrr_list = np.nan_to_num(Metrics.mrr_at_k(pred_scores[:, body_item], ground_truth[:, body_item], k)).tolist()
    body_ndcg_list = np.nan_to_num(Metrics.ndcg_at_k(pred_scores[:, body_item], ground_truth[:, body_item], k)).tolist()
    body_hr_list = np.nan_to_num(Metrics.hit_at_k(pred_scores[:, body_item], ground_truth[:, body_item], k)).tolist()

    return np.mean(head_ndcg_list), np.mean(head_recall_list), np.mean(tail_ndcg_list), np.mean(tail_recall_list), np.mean(body_ndcg_list), np.mean(body_recall_list)
