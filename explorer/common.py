from onnx import mapping


def elem_type_to_name(elem_type: int):
    type_map = {
        1: 'float32',
        2: 'uint8',
        3: 'int8',
        4: 'uint16',
        5: 'int16',
        6: 'int32',
        7: 'int64',
        8: 'string',
        9: 'boolean',
        10: 'float16',
        11: 'float64',
        12: 'uint32',
        13: 'uint64',
        14: '',
        15: 'float32',
        16: 'float32',
    }
    type_map = mapping.TENSOR_TYPE_TO_NP_TYPE
    if elem_type in type_map.keys():
        return type_map[elem_type].name
    else:
        return 'undefined'