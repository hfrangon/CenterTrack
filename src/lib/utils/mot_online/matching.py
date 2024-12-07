from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lap
import numpy as np
import scipy
import torch
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.spatial.distance import cdist
from torch import nn




chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(tracks, detections, iou_thresh,is_low=False):
    cost_matrix = iou_distance(tracks, detections)
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost_matrix=1-cost_matrix
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=iou_thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])# 返回的是匹配的索引 ix代表行，mx代表列 表示第ix行和第mx列匹配
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]
    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    return ious(atlbrs, btlbrs)

def hm_iou_distance(atracks, btracks):
    iou_cost_matrix = iou_distance(atracks, btracks)
    if iou_cost_matrix.size == 0:
        return iou_cost_matrix

    bboxes1 =np.array([track.tlbr for track in atracks])
    bboxes2 =np.array([track.tlbr for track in btracks])
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    yy11 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    yy12 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

    yy21 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    yy22 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    o = (yy12 - yy11) / (yy22 - yy21)
    return iou_cost_matrix * o

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix

def embedding_distance2(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    track_features = np.asarray([track.features[0] for track in tracks], dtype=np.float64)
    cost_matrix2 = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    track_features = np.asarray([track.features[len(track.features)-1] for track in tracks], dtype=float)
    cost_matrix3 = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    for row in range(len(cost_matrix)):
        cost_matrix[row] = (cost_matrix[row]+cost_matrix2[row]+cost_matrix3[row])/3
    return cost_matrix


def vis_id_feature_A_distance(tracks, detections, metric='cosine'):
    track_features = []
    det_features = []
    leg1 = len(tracks)
    leg2 = len(detections)
    cost_matrix = np.zeros((leg1, leg2), dtype=np.float64)
    cost_matrix_det = np.zeros((leg1, leg2), dtype=np.float64)
    cost_matrix_track = np.zeros((leg1, leg2), dtype=np.float64)
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    if leg2 != 0:
        cost_matrix_det = np.maximum(0.0, cdist(det_features, det_features, metric))
    if leg1 != 0:
        cost_matrix_track = np.maximum(0.0, cdist(track_features, track_features, metric))
    if cost_matrix.size == 0:
        return track_features, det_features, cost_matrix, cost_matrix_det, cost_matrix_track
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    if leg1 > 10:
        leg1 = 10
        tracks = tracks[:10]
    if leg2 > 10:
        leg2 = 10
        detections = detections[:10]
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    return track_features, det_features, cost_matrix, cost_matrix_det, cost_matrix_track

def gate_cost_matrix(kf,cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix

def mds_distance(tracks, detections):

    maha_cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    mask = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    gating_threshold = chi2inv95[4]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = track.kalman_filter.gating_distance(
            track.mean, track.covariance,measurements,only_position=False)
        maha_cost_matrix[row,:] = gating_distance
        mask[row,gating_distance > gating_threshold] = 1
        maha_cost_matrix[row,gating_distance > gating_threshold] = gating_threshold
    maha_cost_matrix = gating_threshold - maha_cost_matrix
    for row, track in enumerate(tracks):
        maha_cost_matrix[row,:] = nn.functional.softmax(torch.tensor(maha_cost_matrix[row,:]),dim=0).numpy()
    maha_cost_matrix[mask==1] = 0
    return maha_cost_matrix

def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def offset_distance(tracks, detections):


    track_coords = np.array([track.offset for track in tracks])  # (m, d)
    detection_coords = np.array([detection.offset for detection in detections])  # (n, d)

    # Step 1: 归一化矩阵
    track_norms = np.linalg.norm(track_coords, axis=1, keepdims=True)  # (m, 1)
    detection_norms = np.linalg.norm(detection_coords, axis=1, keepdims=True)  # (n, 1)

    track_normalized = track_coords / track_norms  # (m, d)
    detection_normalized = detection_coords / detection_norms  # (n, d)

    # Step 2: 计算余弦相似度矩阵
    direction_cost_matrix = np.dot(track_normalized, detection_normalized.T)  # (m, n)
    direction_cost_matrix = np.clip(direction_cost_matrix, -1.0, 1.0)
    direction_cost_matrix = (np.pi/2-np.arccos(direction_cost_matrix))/np.pi
    return direction_cost_matrix


def assignment(cost_matrix, thresh=0.):
    try:  # [hgx0411] goes here!
        import lap
        if thresh != 0:
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        else:
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def association(tracks, detections, iou_thresh,is_low=False):
    iou_cost_matrix = hm_iou_distance(tracks, detections)
    iou_thresh=1-iou_thresh
    if iou_cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(iou_cost_matrix.shape[0])), tuple(range(iou_cost_matrix.shape[1]))
    maha_cost_matrix = mds_distance(tracks, detections)

    if min(iou_cost_matrix.shape) > 0:
        a = (iou_cost_matrix > iou_thresh).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            cost_matrix = iou_cost_matrix + 0.02 * maha_cost_matrix
            matched_indices =assignment(-cost_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))


    unmatched_trackers = []
    for t, trk in enumerate(tracks):
        if t not in matched_indices[:, 0]:
            unmatched_trackers.append(t)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 1]:
            unmatched_detections.append(d)

    direction_cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if not is_low:
        direction_cost_matrix = offset_distance(tracks, detections)
    #direction_cost_matrix = offset_distance(tracks, detections)

    # filter out matched
    matches = []
    for m in matched_indices:
        if iou_cost_matrix[m[0], m[1]]<iou_thresh or maha_cost_matrix[m[0], m[1]]==0 or direction_cost_matrix[m[0], m[1]]<0:
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if len(matches)==0:
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_trackers),np.array(unmatched_detections)


def first_association(tracks, detections, iou_thresh):
    score_matrix = np.asarray([track.score for track in tracks], dtype=np.float32).reshape(-1,1)
    iou_cost_matrix = hm_iou_distance(tracks, detections)
    iou_cost_matrix= iou_cost_matrix*score_matrix
    iou_matrix = iou_distance(tracks, detections)
    iou_thresh = 1 - iou_thresh
    if iou_cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(iou_cost_matrix.shape[0])), tuple(
            range(iou_cost_matrix.shape[1]))
    maha_cost_matrix = mds_distance(tracks, detections)*score_matrix

    if min(iou_cost_matrix.shape) > 0:
        a = (iou_matrix > iou_thresh).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            cost_matrix = iou_cost_matrix + 0.2 * maha_cost_matrix
            matched_indices = assignment(-cost_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_trackers = []
    for t, trk in enumerate(tracks):
        if t not in matched_indices[:, 0]:
            unmatched_trackers.append(t)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 1]:
            unmatched_detections.append(d)

    direction_cost_matrix = offset_distance(tracks, detections)
    # direction_cost_matrix = offset_distance(tracks, detections)

    # filter out matched
    matches = []
    for m in matched_indices:
        if iou_cost_matrix[m[0], m[1]] < iou_thresh or maha_cost_matrix[m[0], m[1]] == 0 or direction_cost_matrix[
            m[0], m[1]] < 0:
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_trackers), np.array(unmatched_detections)
