import numpy as np
import vg
from pytransform3d.rotations import matrix_from_axis_angle


def _rotmat(vector, points):
    """
    Rotates a 3xn array of 3D coordinates from the +z normal to an
    arbitrary new normal vector.
    """

    vector = vg.normalize(vector)
    axis = vg.perpendicular(vg.basis.z, vector)
    angle = vg.angle(vg.basis.z, vector, units='rad')

    a = np.hstack((axis, (angle,)))
    R = matrix_from_axis_angle(a)

    r = R.from_matrix(R)
    rotmat = r.apply(points)

    return rotmat