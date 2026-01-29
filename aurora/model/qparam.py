import math

class QParams:

    group_size: dict
    bits: list
    bits_prop: list
    scale_bits: int

    desc: str

    def __init__(self, group_size, bits, bits_prop, scale_bits=4, scale_groups=32):

        self.bits = bits
        self.bits_prop = bits_prop
        self.scale_bits = scale_bits
        self.scale_groups = scale_groups
        # Allow group size per bitrate

        if isinstance(group_size, dict):
            self.group_size = { int(b): g for b, g in group_size.items() }
        elif isinstance(group_size, list):
            assert len(group_size) == len(bits)
            self.group_size = { b: g for g, b in zip(group_size, bits) }
        else:
            self.group_size = { b: group_size for b in bits }

        self.desc = self.get_desc()


    def __repr__(self):

        if len(set(self.group_size.values())) == 1:
            _gs = str(list(self.group_size.values())[0])
        else:
            _gs = "[" + ", ".join(str(self.group_size[b]) for b in self.bits) + "]"
        _b = "[" + ", ".join(str(b) for b in self.bits) + "]"
        _bp = "[" + ", ".join(str(bp) for bp in self.bits_prop) + "]"
        _sb = "4"
        return "QParams(" + _gs + ", " + _b + ", " + _bp + ", " + _sb + ")"


    def get_dict(self):

        return { "group_size": self.group_size,
                 "bits": self.bits,
                 "bits_prop": self.bits_prop,
                 "scale_bits": self.scale_bits}


    @staticmethod
    def from_dict(qp_dict):

        return QParams(qp_dict["group_size"],
                       qp_dict["bits"],
                       qp_dict["bits_prop"],
                       qp_dict["scale_bits"])


    def total_bits(self, shape):

        rows = shape[0]
        columns = shape[1]
        numel = rows * columns

        groups = 0
        remaining_columns = columns
        bits_groups = []
        for b, p in zip(self.bits, self.bits_prop):
            gsz = self.group_size[b]
            g = math.ceil(min(columns * p, remaining_columns) / gsz)
            groups += g
            remaining_columns -= g * gsz
            bits_groups.append(g)

        assert remaining_columns <= 0

        total_bits = 0
        #tr = rows
        tc = columns

        for g, b in zip(bits_groups, self.bits):

            c = self.group_size[b] * g
            r = rows
            if c > tc: c = tc
            tc -= c
            total_bits += r * c * b                         # q_weight

        total_bits += groups * 16                           # q_scale_max
        total_bits += groups * (16 + 16)                    # q_groups
        total_bits += groups * rows * self.scale_bits    # q_scale
        total_bits ++ columns * 32                             # q_invperm

        return total_bits


    def bpw(self, shape):

        rows = shape[0]
        columns = shape[1]
        numel = rows * columns

        return self.total_bits(shape) / numel


    def get_desc(self, filename = False):
        #得到当前QParams对应的描述信息，比如"0.05:3b_64g/0.95:2b_64g s4"
        s = ""
        for b, p in zip(self.bits, self.bits_prop):
            if s != "": s += ("__" if filename else "/")
            g = self.group_size[b]
            s += f"{p}{'___' if filename else ':'}{b}b_{g}g"

        s += f" s{self.scale_bits}"

        return s

