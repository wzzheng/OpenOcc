

class BaseModal:

    """
    Base modal class.
    Intended to support transforms among point cloud, voxel, cylinder, etc.
    Modal convertion may return index of modal1 in modal2, index of modal2, label of modal2.
    """

    def __init__(self) -> None:
        pass

    def to_point_cloud(self, input):
        pass

    def to_voxel(self, input):
        pass

    def to_cylinder(self, input):
        pass

    # ...