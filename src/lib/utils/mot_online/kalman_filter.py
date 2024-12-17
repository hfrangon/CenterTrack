# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
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


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self.A = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self.A[i, ndim + i] = dt # A矩阵
        self.H = np.eye(ndim, 2 * ndim)# H矩阵

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20 # sigmaP（过程噪声的标准差）
        self._std_weight_velocity = 1. / 160 # sigmaV (观测噪声的标准差)

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measure vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 measure.

        """
        # 初始化有哪些 A H Q R 矩阵 x0 P0
        mean_pos = measurement # x y a h 观测值(z)
        mean_vel = np.zeros_like(mean_pos)
        # mean_vel[0] = velocity[0]
        # mean_vel[1] = velocity[1]
        measure = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return measure, covariance

    def predict(self, measure, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        measure : ndarray
            The 8 dimensional measure vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measure vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 measure.

        """
        std_pos = [
            self._std_weight_position * measure[3],
            self._std_weight_position * measure[3],
            1e-2,
            self._std_weight_position * measure[3]]
        std_vel = [
            self._std_weight_velocity * measure[3],
            self._std_weight_velocity * measure[3],
            1e-5,
            self._std_weight_velocity * measure[3]]
        q = np.diag(np.square(np.r_[std_pos, std_vel]))

        #measure = np.dot(self.A, measure)
        X_prior = np.dot(measure, self.A.T)
        P_prior = np.linalg.multi_dot((
            self.A, covariance, self.A.T)) + q

        return X_prior, P_prior

    @staticmethod
    def mapping(x):
        # 1-(-(x-0.4)*(x-1)+1)*x
        # 1 - 0.5 * (1 + np.sin(np.pi * (x - 0.5)))
        return 1-(-(x-0.6)*(x-1)+1)*x

    def project(self, measure, covariance,confidence=.0):
        """Project state distribution to measurement space.
            Z = H*X + v
        Parameters
        ----------
        measure : ndarray
            The state's measure vector (8 dimensional array).(x y a h vx vy va vh)
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        confidence: 检测置信度
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected measure and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * measure[3],
            self._std_weight_position * measure[3],
            1e-1,
            self._std_weight_position * measure[3]]
        std = [KalmanFilter.mapping(confidence)*x for x in std]
        #std =[(1-confidence)*x for x in std]
        R = np.diag(np.square(std))

        Z = np.dot(self.H, measure)
        projected_cov =np.linalg.multi_dot((
            self.H, covariance, self.H.T)) + R
        return Z, projected_cov

    def multi_predict(self, measure, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        measure : ndarray
            The Nx8 dimensional measure matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the measure vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 measure.
        """
        std_pos = [
            self._std_weight_position * measure[:, 3],
            self._std_weight_position * measure[:, 3],
            1e-2 * np.ones_like(measure[:, 3]),
            self._std_weight_position * measure[:, 3]]
        std_vel = [
            self._std_weight_velocity * measure[:, 3],
            self._std_weight_velocity * measure[:, 3],
            1e-5 * np.ones_like(measure[:, 3]),
            self._std_weight_velocity * measure[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        Q = []
        for i in range(len(measure)):
            Q.append(np.diag(sqr[i]))
        Q = np.asarray(Q)

        X_prior = np.dot(measure, self.A.T)
        left = np.dot(self.A, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self.A.T) + Q

        return X_prior, covariance

    def update(self, measure, covariance, measurement, confidence =.0):
        """Run Kalman filter correction step.

        Parameters
        ----------
        measure : ndarray 先验值
            The predicted state's measure vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray 观测值
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        Args:
            confidence:
            confidence:

        """
        # measure
        #
        projected_mean, projected_cov = self.project(measure, covariance,confidence)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self.H.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        X_post = measure + np.dot(innovation, kalman_gain.T)
        P_post = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return X_post, P_post


    def gating_distance(self,measure, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        measure : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (measure, covariance) and
            `measurements[i]`.
        """
        measure, covariance = self.project(measure, covariance)
        if only_position:
            measure, covariance = measure[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - measure
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')

