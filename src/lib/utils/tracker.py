from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
import copy
from .mot_online.kalman_filter import KalmanFilter
from .mot_online.basetrack import BaseTrack, TrackState
from .mot_online import matching


class STrack(BaseTrack):
    '''
        实际用于进行卡尔曼状态预测的类
        包括预测、更新、激活等方法
    '''

    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                #跟踪丢失的情况下 不是直接使用先验值进行预测 而是将对应的h速度置为0
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:  #只有在初始化第一帧的时候才会直接激活 之后的帧都是通过update激活 会被加入到unconfirmed中
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh),new_track.score
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        # todo 在这里改变score的值 用于给追踪也增加置信度 从而实现分级匹配
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
    Update a matched track
    :type new_track: STrack
    :type frame_id: int
    :type update_feature: bool
    :return:
    """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh), new_track.score)
        self.state = TrackState.Tracked
        self.is_activated = True

        # todo 在这里改变score的值 用于给追踪也增加置信度 从而实现分级匹配
        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
            width, height)`.
    """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box (top left x ,top left y, w,h)to format `(min x, min y, max x, max y)`, i.e.,
    `(top left, bottom right)`.
    """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.args = args
        self.det_thresh = args.new_thresh  # 0.3
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.reset()

    # below has no effect to final output, just to be compatible to codebase
    def init_track(self, results):
        for item in results:
            if item['score'] > self.opt.new_thresh and item['class'] == 1:
                self.id_count += 1
                item['active'] = 1
                item['age'] = 1
                item['tracking_id'] = self.id_count
                if not ('ct' in item):
                    bbox = item['bbox']
                    item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                self.tracks.append(item)

    def reset(self):
        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.tracks = []

        # below has no effect to final output, just to be compatible to codebase
        self.id_count = 0

    def step(self, results, public_det=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        detections = []
        detections_second = []

        scores = np.array([item['score'] for item in results if item['class'] == 1], np.float32)
        bboxes = np.vstack([item['bbox'] for item in results if item['class'] == 1])  # N x 4, x1y1x2y2

        remain_inds = scores >= self.args.track_thresh  # boolean array
        dets = bboxes[remain_inds]  # 使用boolean array进行过滤 只保留大于阈值的检测结果
        scores_keep = scores[remain_inds]  # score为检测结果的置信度

        inds_low = scores > self.args.out_thresh
        inds_high = scores < self.args.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]# 低置信度的检测结果 第二次在进行匹配
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:  # 用于初次加入到tracked_stracks中的track 即才新生不久的追踪轨迹
                unconfirmed.append(track)  # 未激活的track 不参与本次匹配
            else:
                tracked_stracks.append(track)
        #track_id是通过next_id()函数唯一生成的 lost_stracks是指短暂丢失的track(仍旧存活) removed_stracks是指被移除的track
        ''' Step 2: First association, with Kalman and IOU'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)  # 按照track_id进行合并
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)  # Strack结构内包含mean covariance 用于KF预测
        dists = matching.iou_distance(strack_pool, detections)  # 和高分检测结果进行匹配 通过iou_distance计算距离
        # dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        # 返回的matches是一个二维数组，每一行代表一个匹配的结果，第一列代表strack_pool的索引，第二列代表detections的索引
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)  # 更新track的mean covariance score为当前帧的检测置信度
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)  # 从lost_stracks中找到的track

        ''' Step 3: Second association, association the untrack to the low score detections， with IOU'''
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        # 第二次匹配时 不会匹配已丢失的track(lost_stracks) 即Lost_tracks 只第一次和高分的检测结果进行匹配
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            # if track.state == TrackState.Tracked:
            #     track.update(det, self.frame_id)
            #     activated_starcks.append(track)
            # else:
            #     track.re_activate(det, self.frame_id, new_id=False)# 标记为lost的重新被激活
            #     refind_stracks.append(track)
            track.update(det, self.frame_id)
            activated_starcks.append(track)

        # 将第二次还是未匹配的track加入到lost_stracks中 下次匹配时会进行处理
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # 不是直接重新生成新的track 而是将未匹配的检和unconfirmed中的track进行匹配
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]# 第一次高分未匹配上的检测结果
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)  # todo 这个阈值是否可以设置更高
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        # TODO 为什么这里直接就移除了而不是再多保留一段时间
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        ret = []
        for track in output_stracks:
            track_dict = {}
            track_dict['score'] = track.score
            track_dict['bbox'] = track.tlbr
            bbox = track_dict['bbox']
            track_dict['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            track_dict['active'] = 1 if track.is_activated else 0
            track_dict['tracking_id'] = track.track_id
            track_dict['class'] = 1
            ret.append(track_dict)

        self.tracks = ret
        return ret


def joint_stracks(tlista, tlistb):
    "'合并两个strack列表'"
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


def remove_fp_stracks(stracksa, n_frame=10):
    remain = []
    for t in stracksa:
        score_5 = t.score_list[-n_frame:]
        score_5 = np.array(score_5, dtype=np.float32)
        index = score_5 < 0.45
        num = np.sum(index)
        if num < n_frame:
            remain.append(t)
    return remain
