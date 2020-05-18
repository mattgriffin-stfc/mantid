# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2020 ISIS Rutherford Appleton Laboratory UKRI,
#     NScD Oak Ridge National Laboratory, European Spallation Source
#     & Institut Laue - Langevin
# SPDX - License - Identifier: GPL - 3.0 +
#  This file is part of the mantid workbench.

# 3rdparty imports
from matplotlib.path import Path
from matplotlib.patches import Circle, Ellipse, Patch, PathPatch, Wedge
from matplotlib.transforms import Affine2D, IdentityTransform
import numpy as np


class EllipticalShell(Patch):
    """
    Elliptical shell patch.
    """

    def __str__(self):
        return f"EllipticalShell(center={self.center}, width={self.width}, height={self.height}, thick={self.thick}, angle={self.angle})"

    def __init__(self, center, width, height, thick, angle=0.0, **kwargs):
        """
        Draw an elliptical ring centered at *x*, *y* center with outer width (horizontal diameter)
        *width* and outer height (vertical diameter) *height* with a ring thickness of *thick*
        Valid kwargs are:
        %(Patch)s
        """
        super().__init__(**kwargs)
        self.center = center
        self.height, self.width = height, width
        self.thick = thick
        self.angle = angle
        self._recompute_path()
        # Note: This cannot be calculated until this is added to an Axes
        self._patch_transform = IdentityTransform()

    def _recompute_path(self):
        # Form the outer ring
        arc = Path.arc(theta1=0.0, theta2=360.0)
        # Draw the outer unit circle followed by a reversed and scaled inner circle
        v1 = arc.vertices
        v2 = arc.vertices[::-1] * float(1.0 - self.thick)
        v = np.vstack([v1, v2, v1[0, :], (0, 0)])
        c = np.hstack([arc.codes, arc.codes, Path.MOVETO, Path.CLOSEPOLY])
        c[len(arc.codes)] = Path.MOVETO
        # Final shape acheieved through axis transformation. See _recompute_transform
        self._path = Path(v, c)

    def _recompute_transform(self):
        """NOTE: This cannot be called until after this has been added
                 to an Axes, otherwise unit conversion will fail. This
                 makes it very important to call the accessor method and
                 not directly access the transformation member variable.
        """
        center = (self.convert_xunits(self.center[0]), self.convert_yunits(self.center[1]))
        width = self.convert_xunits(self.width)
        height = self.convert_yunits(self.height)
        self._patch_transform = Affine2D() \
            .scale(width * 0.5, height * 0.5) \
            .rotate_deg(self.angle) \
            .translate(*center)

    def get_patch_transform(self):
        self._recompute_transform()
        return self._patch_transform

    def get_path(self):
        if self._path is None:
            self._recompute_path()
        return self._path


class MplPainter(object):
    """
    Implementation of a PeakPainter that uses matplotlib to draw
    """
    ZOOM_PAD_FRAC = 0.2

    def __init__(self, axes):
        """
        :param axes: A matplotlib.axes.Axes instance to draw on
        """
        if not hasattr(axes, "scatter"):
            raise ValueError("Expected matplotlib.axes.Axes instance. Found {}.".format(type(axes)))
        self._axes = axes

    @property
    def axes(self):
        return self._axes

    def remove(self, artist):
        """
        Remove the painted artifact from the drawn destination
        :param artists: The Artists drawn on the axes
        """
        try:
            artist.remove()
        except ValueError:
            # May have already been removed by a figure/axes clear
            pass

    def circle(self, x, y, radius, **kwargs):
        """Draw a circle on the Axes
        :param x: X coordinate of the center
        :param y: Y coordinate of the center
        :param radius: Radius of the circle
        :param kwargs: Additional matplotlib properties to pass to the call
        """
        return self._axes.add_patch(Circle((x, y), radius, **kwargs))

    def cross(self, x, y, half_width, **kwargs):
        """Draw a cross at the given location
        :param x: X coordinate of the center
        :param y: Y coordinate of the center
        :param half_width: Half-width of cross
        :param kwargs: Additional matplotlib properties to pass to the call
        """
        verts = ((x - half_width, y + half_width), (x + half_width, y - half_width),
                 (x + half_width, y + half_width), (x - half_width, y - half_width))
        codes = (Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO)
        return self._axes.add_patch(PathPatch(Path(verts, codes), **kwargs))

    def ellipse(self, x, y, width, height, angle=0.0, **kwargs):
        """Draw an ellipse at the given location
        :param x: X coordinate of the center
        :param y: Y coordinate of the center
        :param width: Size in X dimension
        :param height: Size in Y dimension
        :param angle: Angle in degrees w.r.t X axis
        :param kwargs: Additional matplotlib properties to pass to the call
        """
        return self._axes.add_patch(Ellipse((x, y), width, height, angle, **kwargs))

    def elliptical_shell(self, x, y, outer_width, outer_height, thick, angle=0.0, **kwargs):
        """Draw an ellipse at the given location
        :param x: X coordinate of the center
        :param y: Y coordinate of the center
        :param outer_width: Size in X dimension of outer ellipse
        :param height: Size in Y dimension of outer ellipse
        :param thick: Thickness of shell
        :param angle: Angle in degrees w.r.t X axis
        :param kwargs: Additional matplotlib properties to pass to the call
        """
        return self._axes.add_patch(
            EllipticalShell((x, y), outer_width, outer_height, thick, angle, **kwargs))

    def shell(self, x, y, outer_radius, thick, **kwargs):
        """Draw a wedge on the Axes
        :param x: X coordinate of the center
        :param y: Y coordinate of the center
        :param outer_radius: Radius of the circle outer edge of the shell
        :param thick: The thickness of the shell
        :param kwargs: Additional matplotlib properties to pass to the call
        """
        return self._axes.add_patch(
            Wedge((x, y), outer_radius, theta1=0.0, theta2=360., width=thick, **kwargs))

    def zoom_to(self, artist):
        """Set the view such that the given artist is in the center
        """
        # Use the bounding box of the artist with small amount of padding
        # to set the axis limits
        to_data_coords = self._axes.transData.inverted()
        artist_bbox = artist.get_extents()
        ll, ur = to_data_coords.transform(artist_bbox.min), \
            to_data_coords.transform(artist_bbox.max)
        # pad by fraction of maximum width so the artist is still in the center
        xl, xr = ll[0], ur[0]
        yb, yt = ll[1], ur[1]
        padding = max(xr - xl, yt - yb) * self.ZOOM_PAD_FRAC
        self._axes.set_xlim(xl - padding, xr + padding)
        self._axes.set_ylim(yb - padding, yt + padding)


class Painted(object):
    """Combine a collection of artists with the painter that created them"""

    def __init__(self, painter, artists):
        self._painter = painter
        self._artists = artists

    @property
    def artists(self):
        return self._artists

    @property
    def painter(self):
        return self._painter

    def remove(self):
        for artist in self._artists:
            self._painter.remove(artist)

    def zoom_to(self):
        """
        Place the painted objects at the center of the view.
        There is an assumption that the final artist represents the "largest"
        object painted on the screen
        """
        self._painter.zoom_to(self._artists[-1])
