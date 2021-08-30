import matplotlib
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.path import Path
from matplotlib.ticker import NullLocator, Formatter, FixedLocator
from matplotlib.transforms import Affine2D, BboxTransformTo, Transform
from matplotlib.projections import register_projection
import matplotlib.spines as mspines
import matplotlib.axis as maxis
import numpy as np

from custom_projections import GeoAxes

class HealpixAxes(GeoAxes):
    name = 'healpix'

    class HealpixTransform(Transform):
        input_dims = output_dims = 2

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform_non_affine(self, ll):
            longitude, latitude = ll.T

            # Pre-compute some values
            H = 4
            K = 3
            theta_x = np.arcsin((K-1)/K)

            x = longitude
            y = np.pi/2*K/H*np.sin(latitude)

            cap = (latitude > theta_x)
            base = (latitude < -theta_x)
            poles = (cap | base)
            sigma = np.sqrt(K*(1-np.abs(np.sin(latitude))))
            if (K % 2 == 0):
              omega = 0
            else:
              omega = 1
            y[cap] = np.pi/H*(2-sigma[cap])
            y[base] = -np.pi/H*(2-sigma[base])

            phi_c = -np.pi + (2*np.floor((longitude+np.pi)*H/(2*np.pi)+(1-omega)/2) + omega)*np.pi/H

            poles = base
            x[poles] = phi_c[poles] + (longitude[poles]-phi_c[poles])*sigma[poles]
            poles = cap
            x[poles] = phi_c[poles] + (longitude[poles]-phi_c[poles])*sigma[poles]

            return np.column_stack([x, y])

        def transform_path_non_affine(self, path):
            # vertices = path.vertices
            ipath = path.interpolated(self._resolution)
            return Path(self.transform(ipath.vertices), ipath.codes)

        def inverted(self):
            return HealpixAxes.InvertedHealpixTransform(self._resolution)

    class InvertedHealpixTransform(Transform):
        input_dims = output_dims = 2

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform_non_affine(self, xy):
            x, y = xy.T
            H = 4
            K = 3
            if (K % 2 == 0):
              omega = 0
            else:
              omega = 1
            alpha = 3*np.pi/(2*H)

            y_x = np.pi/2*(K-1)/H
            cap = (y > y_x)
            base = (y < y_x)
            poles = (cap | base)

            longitude = x
            latitude = np.zeros_like(y)
            latitude[~poles] = np.arcsin(y[~poles]/alpha)

            sigma = (K+1)/2 - np.abs(y*H)/np.pi
            x_c = -np.pi + (2*np.floor((x+np.pi)*H/(2*np.pi) + (1-omega)/2) + omega)*np.pi/H

            longitude[poles] = (x_c + (x - x_c)/sigma**0.5)[poles]

            latitude[cap] = np.arcsin(1-sigma[cap]/K)
            latitude[base] = -np.arcsin(1-sigma[base]/K)

            return np.column_stack([longitude, latitude])

        def inverted(self):
            return HealpixAxes.HealpixTransform(self._resolution)

    def __init__(self, *args, **kwargs):
        self._longitude_cap = np.pi / 2.0
        super().__init__(*args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.cla()

    def _get_core_transform(self, resolution):
        return self.HealpixTransform(resolution)

    def _gen_axes_spines(self):
        x = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,
                      2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25, 0, 0])/2
        y = np.array([0.75, 1, 0.75, 1, 0.75, 1, 0.75, 1, 0.75,
                      0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0.75])
        polygon = Polygon(np.array([x,y]).T)

        path = polygon.get_path()
        spine = mspines.Spine(axes=self, spine_type='circle', path=path)
        spine.set_transform(self.transAxes)
        return {'geo': spine}

    def _gen_axes_patch(self):
        """
        Override this method to define the shape that is used for the
        background of the plot.  It should be a subclass of Patch.

        In this case, it is a Circle (that may be warped by the axes
        transform into an ellipse).  Any data and gridlines will be
        clipped to this shape.
        """
        x = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,
                      2, 1.75, 1.5, 1.25, 1, 0.75, 0.5, 0.25, 0, 0])/2
        y = np.array([0.75, 1, 0.75, 1, 0.75, 1, 0.75, 1, 0.75,
                      0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0.75])
        return Polygon(np.array([x,y]).T)

register_projection(HealpixAxes)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from healpy.newvisufunc import projview, newprojplot
    import healpy as hp

    lmax = 256
    ell = np.arange(2, lmax)
    Cl = np.zeros(lmax)
    Cl[2:] = 1./ell**2
    np.random.seed(0)
    m1 = hp.synfast(Cl, lmax)


    m2 = np.arange(12)

    ms = [m1, m2]


    for m in ms:
        #projview(m, projection_type='mollweide', graticule=True,
            #graticule_labels=True)
        #projview(m, projection_type='hammer', graticule=True, graticule_labels=True)
        projview(m, projection_type='custom_hammer', graticule=True, graticule_labels=True)
        projview(m, projection_type='healpix', graticule=True, graticule_labels=True)
        plt.show()

