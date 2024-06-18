import numpy as np
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
from abc import ABC


class EvaluationHelper(ABC):

    @classmethod
    def dc_and_assd(cls, result: np.ndarray, reference: np.ndarray):
        dc = EvaluationHelper.dc(result, reference)
        try:
            assd = EvaluationHelper.assd(result, reference)
        except RuntimeError:
            # In case of all-zero sample
            result[-1, -1] = 1
            reference[0, 0] = 1
            assd = EvaluationHelper.assd(result, reference)
        return dc, assd

    @staticmethod
    def dc(result: np.ndarray, reference: np.ndarray) -> float:
        result = np.atleast_1d(result.astype(bool))
        reference = np.atleast_1d(reference.astype(bool))

        intersection = np.count_nonzero(result & reference)

        size_i1 = np.count_nonzero(result)
        size_i2 = np.count_nonzero(reference)

        try:
            dc = 2. * intersection / float(size_i1 + size_i2)
        except ZeroDivisionError:
            dc = 0.0

        return dc

    @classmethod
    def assd(cls,
             result: np.ndarray,
             reference: np.ndarray,
             voxelspacing=None,
             connectivity=1):
    
        assd = np.mean(
            (cls.asd(result, reference, voxelspacing, connectivity),
             cls.asd(reference, result, voxelspacing, connectivity)))
        return assd

    @classmethod
    def asd(cls, result, reference, voxelspacing=None, connectivity=1):

        sds = cls.__surface_distances(result, reference, voxelspacing, connectivity)
        asd = sds.mean()
        return asd

    @staticmethod
    def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
        result = np.atleast_1d(result.astype(bool))
        reference = np.atleast_1d(reference.astype(bool))
        if voxelspacing is not None:
            voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
            voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
            if not voxelspacing.flags.contiguous:
                voxelspacing = voxelspacing.copy()

        # binary structure
        footprint = generate_binary_structure(result.ndim, connectivity)

        # test for emptiness
        if 0 == np.count_nonzero(result):
            raise RuntimeError('The first supplied array does not contain any binary object.')
        if 0 == np.count_nonzero(reference):
            raise RuntimeError('The second supplied array does not contain any binary object.')

        result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
        reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

        dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
        sds = dt[result_border]

        return sds