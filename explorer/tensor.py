import warnings

import numpy as np
import onnx

from .utils import GLOBAL


def create_ndarray_f32(shape):
    return np.ones(shape, dtype=np.float32)


def create_ndarray_int64(shape):
    return np.zeros(shape, dtype=np.int64)


def shape_of_tensor(tensor):
    shape = []
    # for nb in tensor.shape.dim
    for nb in tensor.type.tensor_type.shape.dim:
        if nb.HasField('dim_value'):
            shape.append(nb.dim_value)
        if nb.HasField('dim_param'):
            shape.append(nb.dim_param)
    return shape


def shape_of_initializer(initial):
    shape = []
    # for nb in tensor.shape.dim
    for nb in initial.dims:
        shape.append(nb)
    return shape


def onnxdtype2npdtype(data_type):
    if data_type == onnx.TensorProto.FLOAT:
        return np.float32
    if data_type == onnx.TensorProto.DOUBLE:
        return np.float64
    if data_type == onnx.TensorProto.FLOAT16:
        return np.float16
    if data_type == onnx.TensorProto.INT32:
        return np.int32
    if data_type == onnx.TensorProto.INT16:
        return np.int16
    if data_type == onnx.TensorProto.INT64:
        return np.int64
    if data_type == onnx.TensorProto.INT8:
        return np.int8
    if data_type == onnx.TensorProto.UINT8:
        return np.uint8
    if data_type == onnx.TensorProto.BOOL:
        return np.bool_
    if data_type == onnx.TensorProto.STRING:
        return np.string_


def type_of_tensor(tensor):
    return onnxdtype2npdtype(tensor.type.tensor_type.elem_type)


def npdtype2onnxdtype(npdtype):
    if npdtype == np.float32:
        return onnx.TensorProto.FLOAT
    if npdtype == np.float64:
        return onnx.TensorProto.DOUBLE
    if npdtype == np.float16:
        return onnx.TensorProto.FLOAT16
    if npdtype == np.int32:
        return onnx.TensorProto.INT32
    if npdtype == np.int16:
        return onnx.TensorProto.INT16
    if npdtype == np.int64:
        return onnx.TensorProto.INT64
    if npdtype == np.int8:
        return onnx.TensorProto.INT8
    if npdtype == np.uint8:
        return onnx.TensorProto.UINT8
    if npdtype == np.bool_:
        return onnx.TensorProto.BOOL
    if npdtype == np.bytes_:
        return onnx.TensorProto.STRING
    if npdtype.type == np.string_:
        return onnx.TensorProto.STRING


def tensorproto2ndarray(initial):
    shape = shape_of_initializer(initial)
    ndtype = onnxdtype2npdtype(initial.data_type)
    if initial.raw_data == b'':
        arr = np.zeros(shape, ndtype).reshape((-1))
        if ndtype == np.float32:
            arr = np.fromiter(initial.float_data, dtype=ndtype)

        elif ndtype == np.int32:
            arr = np.fromiter(initial.int32_data, dtype=ndtype)

        elif ndtype == np.float16:
            raw = list(initial.int32_data)
            raw = np.fromiter(raw, dtype=np.uint16)
            mem = raw.tobytes()
            arr = np.frombuffer(mem, dtype=np.float16).reshape(shape)

        elif ndtype == np.int64:
            arr = np.fromiter(initial.int64_data, dtype=ndtype)

        elif ndtype == np.float64:
            arr = np.fromiter(initial.double_data, dtype=ndtype)

        elif ndtype == np.string_:
            arr = np.array(initial.string_data, dtype=ndtype)
    else:
        arr = np.frombuffer(initial.raw_data, dtype=ndtype)
    # if len(shape):
    arr = arr.reshape(shape)
    return arr


def get_attribute_data(att):
    if att.type == att.INTS:
        val = []
        for ints in att.ints:
            val.append(ints)
        return val
    elif att.type == att.INT:
        return att.i
    elif att.type == att.FLOAT:
        return att.f
    elif att.type == att.STRING:
        return att.s
    elif att.type == att.FLOATS:
        val = []
        for f in att.floats:
            val.append(f)
        return val
    elif att.type == att.TENSOR:
        return tensorproto2ndarray(att.t)


def volume(shape: []):
    # if not isinstance(shape,list):
    #     return 1 #scalar
    val = 1 if len(shape) > 0 else 0
    for v in shape:
        val *= v
    return val


def volume_tensor(t):
    if isinstance(t, np.ndarray):
        return volume(t.shape)
    return 1


def narray_calc_sparsity(arr):
    if len(arr.shape) != 2 and len(arr.shape) != 4:
        return 0
    if arr.dtype == np.float32 or arr.dtype == np.float64 or arr.dtype == np.int32 or arr.dtype == np.int8:
        flag = arr == 0
        return flag.sum() / arr.size
    if arr.dtype == np.uint8:
        flag = arr == 128
        return flag.sum() / arr.size
    if arr.dtype == np.float16:
        flag = arr == 0
        return flag.sum() / arr.size
    return 0


def narray_zero_flag(arr):
    if arr.dtype in (np.float32, np.float64, np.int32, np.int8, np.float16):
        flag = arr == 0
    if arr.dtype == np.uint8:
        flag = arr == 128
    return flag


def is_valid_ndarray(x):
    if x is None:
        return False
    if isinstance(x, (list, tuple)) and len(x) == 0:
        return False
    if isinstance(x, np.ndarray):
        if volume(x.shape) == 0:
            return True if x.size else False
        else:
            return True
    return False


def graph_addoutputs(graph: onnx.GraphProto, outnames: [str]) -> onnx.GraphProto:
    tensor_map = GLOBAL['tensor_map']
    for name in outnames:
        if tensor_map is not None and name in tensor_map.keys():
            newout = onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.FLOAT, tensor_map[name].shape)
        else:
            newout = onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.FLOAT, ())
        graph.output.append(newout)
    return graph


def graph_set_inputs(graph: onnx.GraphProto, dynamic_tensors: {}) -> onnx.GraphProto:
    tensor_map = GLOBAL['tensor_map']
    for input in graph.input:
        if dynamic_tensors.keys().__contains__(input.name):
            tensor_map[input.name] = dynamic_tensors[input.name]
            dim = input.type.tensor_type.shape.dim
            for nb, dnb in zip(dim, dynamic_tensors[input.name].shape):
                nb.dim_value = dnb
    return graph


def update_static_tensors(graph: onnx.GraphProto):
    tensor_map = GLOBAL['tensor_map']
    params_map = GLOBAL['params_map']
    for initial in graph.initializer:
        arr = tensorproto2ndarray(initial)
        tensor_map.update({initial.name: arr})

    for node in graph.node:
        if node.op_type == 'Constant':
            for att in node.attribute:
                if att.name == 'value':
                    tensor_map[node.output[0]] = get_attribute_data(att)

    totalparams = 0
    for key in tensor_map.keys():
        params_map[key] = volume(tensor_map[key].shape)
        totalparams += params_map[key]
    GLOBAL['totalparams'] = totalparams


def np_dtype2bytes(ndtype):
    return np.dtype(ndtype).itemsize


def same_shape(shape0, shape1):
    if len(shape1) != len(shape0):
        return False
    for a, b in zip(shape0, shape1):
        if a != b:
            return False
    return True


def search_sparse_blocksize(arr, ratio, deltar_thres=0.1):
    if len(arr.shape) == 2:  # gemm or matmul
        initsize = 2
        validsize = 1
        prevalid0 = True
        prevalid1 = True
        validratio = ratio
        while True:
            # try axis=1
            if prevalid1 and arr.shape[1] % initsize == 0:
                rearr = arr.reshape(arr.shape[0], -1, initsize)
                flag = narray_zero_flag(rearr)
                arrsum = np.sum(flag, -1)
                ratio1 = (arrsum == initsize).sum() / arrsum.size
                if ratio1 > ratio - deltar_thres:
                    valid1 = True
                    validratio = ratio1
                else:
                    valid1 = False
            else:
                valid1 = False

            # try axis=0
            if prevalid0 and arr.shape[0] % initsize == 0:
                rearr = arr.reshape(-1, initsize, arr.shape[1])
                flag = narray_zero_flag(rearr)
                arrsum = np.sum(flag, 1)
                ratio0 = (arrsum == initsize).sum() / arrsum.size
                if ratio0 > ratio - deltar_thres:
                    valid0 = True
                    validratio = ratio0
                else:
                    valid0 = False
            else:
                valid0 = False

            if not valid1 and not valid0:
                break
            validsize = initsize
            initsize *= 2
            prevalid0 = valid0
            prevalid1 = valid1

        # check square
        if prevalid1 and prevalid0:
            rearr = arr.reshape(
                arr.shape[0] // validsize, validsize, arr.shape[1] // validsize, validsize)
            flag = narray_zero_flag(rearr)
            arrsum = np.sum(flag, axis=(1, -1))
            ratios = (arrsum == (validsize * validsize)).sum() / arrsum.size
            if ratios > ratio - deltar_thres:
                return (validsize, validsize), ratios

        return (validsize if prevalid0 else 1, validsize if prevalid1 else 1), validratio

    if len(arr.shape) == 4:  # conv2d
        initsize = 2
        validsize = 1
        prevalid0 = True
        prevalid1 = True
        validratio0 = ratio
        validratio1 = ratio
        while True:
            # try axis=1
            if prevalid1 and arr.shape[1] % initsize == 0:
                rearr = arr.reshape(arr.shape[0], -1, initsize, *arr.shape[2:])
                flag = narray_zero_flag(rearr)
                arrsum = np.sum(flag, 2)
                ratio1 = (arrsum == initsize).sum() / arrsum.size
                if ratio1 > ratio - deltar_thres:
                    valid1 = True
                    validratio1 = ratio1
                else:
                    valid1 = False
            else:
                valid1 = False

            # try axis=0
            if prevalid0 and arr.shape[0] % initsize == 0:
                rearr = arr.reshape(-1, initsize, *arr.shape[1:])
                flag = narray_zero_flag(rearr)
                arrsum = np.sum(flag, 1)
                ratio0 = (arrsum == initsize).sum() / arrsum.size
                if ratio0 > ratio - deltar_thres:
                    valid0 = True
                    validratio0 = ratio0
                else:
                    valid0 = False
            else:
                valid0 = False

            if not valid1 and not valid0:
                break
            validsize = initsize
            initsize *= 2
            prevalid0 = valid0
            prevalid1 = valid1
        # check square
        if validsize > 1 and prevalid1 and prevalid0:
            rearr = arr.reshape(arr.shape[0] // validsize, validsize, arr.shape[1] // validsize, validsize,
                                *arr.shape[2:])
            flag = narray_zero_flag(rearr)
            arrsum = np.sum(flag, axis=(1, 3))
            ratios = (arrsum == (validsize * validsize)).sum() / arrsum.size
            if ratios > ratio - deltar_thres:
                return (validsize, validsize), ratios
        if validratio0 > validratio1:
            return (validsize, 1), validratio0
        return (1, validsize), validratio1

    return (1, 1), ratio


STATIC_TENSOR = 0
DYNAMIC_TENSOR = 1


class Tensor():
    def __init__(self, t):
        from .node import Node
        if isinstance(t, str):
            self.name = t
            self.proto = None
            self.shape = []
            self.np = None
            self.type = DYNAMIC_TENSOR if t != '' else STATIC_TENSOR
            self.dtype = np.float32
        elif isinstance(t, onnx.ValueInfoProto):
            self.name = t.name
            self.proto = t
            self.shape = shape_of_tensor(t)
            self.np = None
            self.type = DYNAMIC_TENSOR
            self.dtype = type_of_tensor(t)
        elif isinstance(t, onnx.TensorProto):
            self.name = t.name
            self.proto = t
            self.np = tensorproto2ndarray(t)
            self.shape = self.np.shape
            self.type = STATIC_TENSOR
            self.dtype = self.np.dtype.type
        else:
            assert 0
        self.sparsity_search()

    def update_tensor(self, data: np.ndarray):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.np = data
        self.update_shape(data.shape)
        self.update_dtype(self.np.dtype.type)

    def update_proto(self, data: np.ndarray):
        self.update_tensor(data)
        self.proto = self.make_tensor_proto()

    def update_shape(self, shape: list):
        if isinstance(shape, np.ndarray):
            assert 0
        self.shape = shape

    def update_dtype(self, dtype):
        self.dtype = dtype

    def shape2str(self):
        st = '['
        for val in self.shape:
            if isinstance(val, str):
                st += val + ','
            else:
                st += str(val) + ','
        st = st[:-1]
        st += ']'
        return st

    def get_shape(self):
        shape = []
        for s in self.shape:
            if isinstance(s, str):
                shape.append(s)
            else:
                shape.append(int(s))
        return shape

    def get_np(self):
        if self.np is not None:
            if same_shape(self.np.shape, self.shape):
                return self.np
        self.np = np.zeros(self.shape, dtype=self.dtype)
        return self.np

    def get_valueorshape(self):
        if self.np is not None:
            return self.np
        return self.shape

    def get_elementsize(self):
        if self.np is None or not isinstance(self.np, np.ndarray):
            return np_dtype2bytes(self.dtype)  # default as float
        return np_dtype2bytes(self.np.dtype)

    def get_memsize(self):
        return volume(self.get_shape()) * self.get_elementsize()

    def sparsity_search(self, thres_size=4096, thres_ratio=0.4):
        if self.type == DYNAMIC_TENSOR:
            self.sparsity = None
            return
        shape = self.get_shape()
        blocksize = (1, 1)
        blockratio = 0
        ratio = 0
        if (volume(shape) > thres_size) and self.np is not None:
            ratio = narray_calc_sparsity(self.np)
            if ratio is not None and ratio > thres_ratio:
                blocksize, blockratio = search_sparse_blocksize(
                    self.np, ratio, deltar_thres=0.1)
        self.sparsity = {'blocksize': blocksize,
                         'blockratio': blockratio, 'ratio': ratio}

    def make_value_proto(self, make_dummy=False):
        shape = self.get_shape()
        if len(self.shape) == 0:
            shape = None
        if self.np is None:
            dtype = npdtype2onnxdtype(self.dtype)
        else:
            dtype = npdtype2onnxdtype(self.np.dtype)
        if self.name == '':
            return None
        # shape = [int(i) for i in shape]
        vinf = onnx.helper.make_tensor_value_info(self.name, dtype, shape)
        return vinf

    def make_tensor_proto(self):
        if self.np is None:
            return None
        if len(self.np.shape) == 0:
            tproto = onnx.helper.make_tensor(self.name, npdtype2onnxdtype(
                self.np.dtype), [], [self.np.item()])
        else:
            if self.np.dtype not in [np.float32, np.int32]:
                raw = True
                data = self.np.tobytes()
            else:
                raw = False
                data = self.np.flatten()
            tproto = onnx.helper.make_tensor(self.name, npdtype2onnxdtype(
                self.np.dtype), self.np.shape, data, raw=raw)
        return tproto


def create_initial_Tensor(name: str, ndarray: np.ndarray):
    t = Tensor(name)
    t.type = STATIC_TENSOR
    t.np = ndarray
    t.shape = t.np.shape
    t.proto = t.make_tensor_proto()
    t.dtype = t.np.dtype.type
    return t
