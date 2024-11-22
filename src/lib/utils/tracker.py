from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
import copy

from .cmc.file import GmcFile
from .mot_online.kalman_filter import KalmanFilter
from .mot_online.basetrack import BaseTrack, TrackState
from .mot_online import matching


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score):

        # wait activate
        self.attr_saved = None
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.time_since_update=0
        self.score = score
        self.tracklet_len = 0
        self.last_observation = None # list of [x1, y1, w, h, frame_id]

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        self.time_since_update +=1

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
                stracks[i].time_since_update += 1

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.last_observation = list(self.tlwh_to_xywh(self._tlwh))+[frame_id]
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        # todo 在这里做oru

        index2 = frame_id
        x2, y2, w2, h2 = STrack.tlwh_to_xywh(new_track.tlwh)
        self.__dict__ = self.attr_saved
        index1 = self.last_observation[4]
        x1, y1, w1, h1 = self.last_observation[:4]
        time_gap = index2 - index1
        dx = (x2 - x1) / time_gap
        dy = (y2 - y1) / time_gap
        dw = (w2 - w1) / time_gap
        dh = (h2 - h1) / time_gap
        for i in range(time_gap):
            x = x1 + (i + 1) * dx
            y = y1 + (i + 1) * dy
            w = w1 + (i + 1) * dw
            h = h1 + (i + 1) * dh
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, np.array([x, y, w / float(h), h])
            )
            if not i == (time_gap - 1):
                self.predict()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update=0
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score =new_track.score

    def calculate_score(self, score):
        return math.exp(-self.time_since_update/5) + (score - 1)

    def update(self, new_track, frame_id,cost=1.0):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :return:

        Args:
            cost: 衡量det和track之间的匹配代价
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.last_observation = list(self.tlwh_to_xywh(self._tlwh))+[frame_id]
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0
        self.score = new_track.score

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)#改变了mean中所有状态
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def camera_update(self, matrix):
        x1, y1, x2, y2 = self.tlbr
        x1_, y1_, _ = matrix @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = matrix @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.mean[:4] = [cx, cy, w / h, h]#只改变了mean的前四个值

    def mark_lost(self):
        """
            Save the parameters before non-observation forward
        """
        self.state = TrackState.Lost
        self.attr_saved = copy.deepcopy(self.__dict__)

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
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret
    @property
    def to_xywh(self):
        ret = self.mean
        ret[2] *= ret[3]
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

    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

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
        self.det_thresh = args.new_thresh
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.reset()


    # below has no effect to final output, just to be compatible to codebase
    def init_track(self, results):
        for item in results:
            if item['score'] > self.args.new_thresh and item['class'] == 1:
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

    def step(self, results, public_det=None,img_info=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []


        scores = np.array([item['score'] for item in results if item['class'] == 1], np.float32)
        bboxes = np.vstack([item['bbox'] for item in results if item['class'] == 1])  # N x 4, x1y1x2y2

        remain_inds = scores >= self.args.track_thresh
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]

        inds_low = scores > self.args.out_thresh
        inds_high = scores < self.args.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
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
            if not track.is_activated:
                unconfirmed.append(track)# todo 为什么unconfirmed 没有predict
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with Kalman and IOU'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # cmc
        # Fix camera motion
        matrix = GmcFile.apply(img_info, self.args.trainval)
        for track in strack_pool:
            track.camera_update(matrix)
        for track in unconfirmed:
            track.camera_update(matrix)


        tracks = [track for track in strack_pool if track.score>= self.args.track_thresh]
        tracks_second =[track for track in strack_pool if track.score < self.args.track_thresh]
        if self.args.use_MDS:
            dists = matching.iou_distance_with_mds(tracks, detections)
        else:
            dists = matching.iou_distance(tracks, detections)
        # dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = tracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id,dists[itracked, idet])
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        """3.1 匹配tracks_second"""
        tracks_second = tracks_second+[tracks[i] for i in u_track]
        u_detection = [detections[i] for i in u_detection]
        if self.args.use_MDS:
            dists = matching.iou_distance_with_mds(tracks_second, u_detection)
        else:
            dists = matching.iou_distance(tracks_second, u_detection)
        matches, u_track, u_detection_s = matching.linear_assignment(dists,thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = tracks_second[itracked]
            det = u_detection[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id,dists[itracked, idet])
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, association the untrack to the low score detections， with IOU'''
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        # todo 使用上一次的检测值做匹配
        r_tracked_stracks = [tracks_second[i] for i in u_track if tracks_second[i].state == TrackState.Tracked]
        if self.args.use_MDS:
            dists = matching.iou_distance_with_mds(r_tracked_stracks, detections_second)
        else:
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id,dists[itracked,idet])
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False,cost=dists[itracked,idet])
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [u_detection[i] for i in u_detection_s]

        if self.args.use_MDS:
            dists = matching.iou_distance_with_mds(unconfirmed, detections)
        else:
            dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id,dists[itracked, idet])
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:#todo 怎么实现匹配几次后再confirm
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

