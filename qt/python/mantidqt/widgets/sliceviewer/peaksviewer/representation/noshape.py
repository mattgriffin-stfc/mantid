# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2020 ISIS Rutherford Appleton Laboratory UKRI,
#     NScD Oak Ridge National Laboratory, European Spallation Source
#     & Institut Laue - Langevin
# SPDX - License - Identifier: GPL - 3.0 +
#  This file is part of the mantid workbench.

# local imports
from .alpha import compute_alpha
from .painter import Painted


class NonIntegratedPeakRepresentation(object):
    """Create a collection of PeakDrawable objects for a non-integrated Peak"""
    VIEW_FRACTION = 0.015

    @classmethod
    def draw(cls, peak_origin, peak_shape, slice_info, painter, fg_color, _):
        """
        Draw the representation of a slice through a peak with no shape
        :param peak_origin: Peak origin in original workspace frame
        :param peak_shape: A reference to the object describing the PeakShape
        :param slice_info: A SliceInfo object detailing the current slice
        :param painter: A reference to a object capable of drawing shapes
        :param fg_color: A str representing the color of the peak shape marker
        :param _: A str representing the color of the background region. Unused
        :return: A new instance of this class
        """
        peak_origin = slice_info.transform(peak_origin)
        x, y, z = peak_origin
        alpha = compute_alpha(z, slice_info.value, slice_info.width)
        painted = None
        if alpha > 0.0:
            effective_radius = slice_info.width * cls.VIEW_FRACTION
            painted = Painted(
                painter, (painter.cross(x, y, effective_radius, alpha=alpha, color=fg_color), ))

        return painted
