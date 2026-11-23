"""cubemap 核心 API（初始未实现）。"""


class CubeMap:
    def to_face_uv(self, d):
        """方向到面/UV"""
        raise NotImplementedError

    def to_dir(self, face, u, v):
        """面/UV 到方向"""
        raise NotImplementedError
