from functools import partial
import numpy as np
import multiprocessing as mp
from mmdet3d.datasets import build_dataset, build_dataloader
import mmcv
from .AP import instance_match, average_precision
import prettytable
from time import time
from functools import cached_property
from shapely.geometry import LineString
from numpy.typing import NDArray
from typing import Dict, List, Optional
from logging import Logger
from mmcv import Config
from copy import deepcopy
from IPython import embed

INTERP_NUM = 200 # number of points to interpolate during evaluation
THRESHOLDS = [0.5, 1.0, 1.5] # AP thresholds
SAMPLE_DIST = 0.15

class VectorEvaluate(object):
    """Evaluator for vectorized map.

    Args:
        dataset_cfg (Config): dataset cfg for gt
        n_workers (int): num workers to parallel
    """

    def __init__(self, dataset_cfg: Config, n_workers: int=8) -> None:
        self.dataset = build_dataset(dataset_cfg)
        self.cat2id = self.dataset.cat2id
        self.id2cat = {v: k for k, v in self.cat2id.items()}
        self.n_workers = n_workers
        self.new_split = 'newsplit' in self.dataset.ann_file
        self.roi_size = self.dataset.roi_size
        if self.roi_size == (60, 30):
            self.thresholds = [0.5, 1.0, 1.5]
        elif self.roi_size == (100, 50):
            self.thresholds = [1.0, 1.5, 2.0]
        
    @cached_property
    def gts(self) -> Dict[str, Dict[int, List[NDArray]]]:
        print('collecting gts...')
        gts = {}
        dataloader = build_dataloader(
            self.dataset, samples_per_gpu=1, workers_per_gpu=4, shuffle=False, dist=False)
        iter_dataloader = iter(dataloader)
        pbar = mmcv.ProgressBar(len(dataloader))
        for data in iter_dataloader:
            token = deepcopy(data['img_metas'].data[0][0]['token'])
            gt = deepcopy(data['vectors'].data[0][0])
            gts[token] = gt
            pbar.update()
            del data # avoid dataloader memory crash
        
        del iter_dataloader
        del dataloader
        return gts
    
    @staticmethod
    def interp_fixed_num_shapely(vector: NDArray, num_pts: int) -> NDArray:
        ''' Interpolate a polyline.
        
        Args:
            vector (array): line coordinates, shape (M, 2)
            num_pts (int): 
        
        Returns:
            sampled_points (array): interpolated coordinates
        '''
        line = LineString(vector)
        distances = np.linspace(0, line.length, num_pts)
        sampled_points = np.array([list(line.interpolate(distance).coords) 
            for distance in distances]).squeeze()
        
        return sampled_points

    @staticmethod
    def interp_fixed_dist(vector: NDArray, sample_dist: float) -> NDArray:
        ''' Interpolate a line at fixed interval.
        
        Args:
            vector (LineString): vector
            sample_dist (float): sample interval
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''
        line = LineString(vector)
        distances = list(np.arange(sample_dist, line.length, sample_dist))
        # make sure to sample at least two points when sample_dist > line.length
        distances = [0,] + distances + [line.length,] 
        
        sampled_points = np.array([list(line.interpolate(distance).coords)
                                for distance in distances]).squeeze()
        
        return sampled_points

    @staticmethod
    def interp_fixed_num(points: NDArray, num_pts: int) -> NDArray:
        """
        Ref: https://github.com/argoverse/av2-api/blob/main/src/av2/geometry/interpolate.py#L130

        Args:
            t: number of points that will be uniformly interpolated and returned
            points: Numpy array of shape (N,2) or (N,3), representing 2d or 3d-coordinates of the arc.

        Returns:
            Numpy array of shape (N,2)

        Raises:
            ValueError: If `points` is not in R^2 or R^3.
        """
        if points.ndim != 2:
            raise ValueError("Input array must be (N,2) or (N,3) in shape.")

        # the number of points on the curve itself
        n, _ = points.shape
        t = num_pts

        # equally spaced in arclength -- the number of points that will be uniformly interpolated
        eq_spaced_points = np.linspace(0, 1, t)

        # Compute the chordal arclength of each segment.
        # Compute differences between each x coord, to get the dx's
        # Do the same to get dy's. Then the hypotenuse length is computed as a norm.
        chordlen: NDArrayFloat = np.linalg.norm(np.diff(points, axis=0), axis=1)  # type: ignore
        # Normalize the arclengths to a unit total
        chordlen = chordlen / np.sum(chordlen)
        # cumulative arclength

        cumarc: NDArray = np.zeros(len(chordlen) + 1)
        cumarc[1:] = np.cumsum(chordlen)

        # which interval did each point fall in, in terms of eq_spaced_points? (bin index)
        tbins: NDArrayInt = np.digitize(eq_spaced_points, bins=cumarc).astype(int)  # type: ignore

        # #catch any problems at the ends
        tbins[np.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1  # type: ignore
        tbins[np.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1

        s = np.divide((eq_spaced_points - cumarc[tbins - 1]), chordlen[tbins - 1])
        anchors = points[tbins - 1, :]
        # broadcast to scale each row of `points` by a different row of s
        offsets = (points[tbins, :] - points[tbins - 1, :]) * s.reshape(-1, 1)
        points_interp: NDArray = anchors + offsets

        return points_interp

    def _evaluate_single(self, 
                         pred_vectors: List, 
                         scores: List, 
                         groundtruth: List, 
                         thresholds: List, 
                         metric: str='metric') -> Dict[int, NDArray]:
        ''' Do single-frame matching for one class.
        
        Args:
            pred_vectors (List): List[vector(ndarray) (different length)], 
            scores (List): List[score(float)]
            groundtruth (List): List of vectors
            thresholds (List): List of thresholds
        
        Returns:
            tp_fp_score_by_thr (Dict): matching results at different thresholds
                e.g. {0.5: (M, 2), 1.0: (M, 2), 1.5: (M, 2)}
        '''

        pred_lines = []

        # interpolate predictions
        for vector in pred_vectors:
            vector = np.array(vector)
            vector_interp = self.interp_fixed_num(vector, INTERP_NUM)
            pred_lines.append(vector_interp)
        if pred_lines:
            pred_lines = np.stack(pred_lines)
        else:
            pred_lines = np.zeros((0, INTERP_NUM, 2))

        # interpolate groundtruth
        gt_lines = []
        for vector in groundtruth:
            vector_interp = self.interp_fixed_num(vector, INTERP_NUM)
            gt_lines.append(vector_interp)
        if gt_lines:
            gt_lines = np.stack(gt_lines)
        else:
            gt_lines = np.zeros((0, INTERP_NUM, 2))
        
        scores = np.array(scores)
        tp_fp_list = instance_match(pred_lines, scores, gt_lines, thresholds, metric) # (M, 2)
        tp_fp_score_by_thr = {}
        for i, thr in enumerate(thresholds):
            tp, fp = tp_fp_list[i]
            tp_fp_score = np.hstack([tp[:, None], fp[:, None], scores[:, None]])
            tp_fp_score_by_thr[thr] = tp_fp_score
        
        return tp_fp_score_by_thr # {0.5: (M, 2), 1.0: (M, 2), 1.5: (M, 2)}
        
    def evaluate(self, 
                 result_path: str, 
                 metric: str='chamfer', 
                 logger: Optional[Logger]=None) -> Dict[str, float]:
        ''' Do evaluation for a submission file and print evalution results to `logger` if specified.
        The submission will be aligned by tokens before evaluation. We use multi-worker to speed up.
        
        Args:
            result_path (str): path to submission file
            metric (str): distance metric. Default: 'chamfer'
            logger (Logger): logger to print evaluation result, Default: None
        
        Returns:
            new_result_dict (Dict): evaluation results. AP by categories.
        '''
        
        results = mmcv.load(result_path)
        results = results['results']
        
        # re-group samples and gt by label
        samples_by_cls = {label: [] for label in self.id2cat.keys()}
        num_gts = {label: 0 for label in self.id2cat.keys()}
        num_preds = {label: 0 for label in self.id2cat.keys()}

        # align by token
        for token, gt in self.gts.items():
            if token in results.keys():
                pred = results[token]
            else:
                pred = {'vectors': [], 'scores': [], 'labels': []}
            
            # for every sample
            vectors_by_cls = {label: [] for label in self.id2cat.keys()}
            scores_by_cls = {label: [] for label in self.id2cat.keys()}

            for i in range(len(pred['labels'])):
                # i-th pred line in sample
                label = pred['labels'][i]
                vector = pred['vectors'][i]
                score = pred['scores'][i]

                vectors_by_cls[label].append(vector)
                scores_by_cls[label].append(score)

            for label in self.id2cat.keys():
                new_sample = (vectors_by_cls[label], scores_by_cls[label], gt[label])
                num_gts[label] += len(gt[label])
                num_preds[label] += len(scores_by_cls[label])
                samples_by_cls[label].append(new_sample)

        result_dict = {}

        print(f'\nevaluating {len(self.id2cat)} categories...')
        start = time()
        if self.n_workers > 0:
            # ctx = mp.get_context("spawn")
            # pool = mp.pool.Pool(self.n_workers, context=ctx)
            pool = mp.Pool(self.n_workers)
        
        sum_mAP = 0
        pbar = mmcv.ProgressBar(len(self.id2cat))
        for label in self.id2cat.keys():
            samples = samples_by_cls[label] # List[(pred_lines, scores, gts)]
            result_dict[self.id2cat[label]] = {
                'num_gts': num_gts[label],
                'num_preds': num_preds[label]
            }
            sum_AP = 0

            fn = partial(self._evaluate_single, thresholds=self.thresholds, metric=metric)
            if self.n_workers > 0:
                tpfp_score_list = pool.starmap(fn, samples)
            else:
                tpfp_score_list = []
                for sample in mmcv.track_iter_progress(samples):
                    tpfp_score_list.append(fn(*sample))
            
            for thr in self.thresholds:
                tp_fp_score = [i[thr] for i in tpfp_score_list]
                tp_fp_score = np.vstack(tp_fp_score) # (num_dets, 3)
                sort_inds = np.argsort(-tp_fp_score[:, -1])

                tp = tp_fp_score[sort_inds, 0] # (num_dets,)
                fp = tp_fp_score[sort_inds, 1] # (num_dets,)
                tp = np.cumsum(tp, axis=0)
                fp = np.cumsum(fp, axis=0)
                eps = np.finfo(np.float32).eps
                recalls = tp / np.maximum(num_gts[label], eps)
                precisions = tp / np.maximum((tp + fp), eps)
                AP = average_precision(recalls, precisions, 'area')
                sum_AP += AP
                result_dict[self.id2cat[label]].update({f'AP@{thr}': AP})

            pbar.update()
            
            AP = sum_AP / len(self.thresholds)
            sum_mAP += AP

            result_dict[self.id2cat[label]].update({f'AP': AP})
        
        if self.n_workers > 0:
            pool.close()
        
        mAP = sum_mAP / len(self.id2cat.keys())
        result_dict.update({'mAP': mAP})
        
        print(f"finished in {time() - start:.2f}s")

        # print results
        table = prettytable.PrettyTable(['category', 'num_preds', 'num_gts'] + 
                [f'AP@{thr}' for thr in self.thresholds] + ['AP'])
        for label in self.id2cat.keys():
            table.add_row([
                self.id2cat[label], 
                result_dict[self.id2cat[label]]['num_preds'],
                result_dict[self.id2cat[label]]['num_gts'],
                *[round(result_dict[self.id2cat[label]][f'AP@{thr}'], 4) for thr in self.thresholds],
                round(result_dict[self.id2cat[label]]['AP'], 4),
            ])
        
        from mmcv.utils import print_log
        print_log('\n'+str(table), logger=logger)
        mAP_normal = 0
        for label in self.id2cat.keys():
            for thr in self.thresholds:
                mAP_normal += result_dict[self.id2cat[label]][f'AP@{thr}']
        mAP_normal = mAP_normal / 9

        print_log(f'mAP_normal = {mAP_normal:.4f}\n', logger=logger)
        # print_log(f'mAP_hard = {mAP_easy:.4f}\n', logger=logger)

        new_result_dict = {}
        for name in self.cat2id:
            new_result_dict[name] = result_dict[name]['AP']

        return new_result_dict