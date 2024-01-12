import onnx
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def onnx_datatype_to_npType(data_type):
    if data_type == 1:
        return np.float32
    else:
        raise TypeError("don't support data type")


def parser_initializer(initializer):
    name = initializer.name
    logging.info(f"initializer name: {name}")

    dims = initializer.dims
    shape = [x for x in dims]
    logging.info(f"initializer with shape:{shape}")

    dtype = initializer.data_type
    logging.info(f"initializer with type: {onnx_datatype_to_npType(dtype)} ")

    # print tenth buffer
    weights = np.frombuffer(initializer.raw_data, dtype=onnx_datatype_to_npType(dtype))
    logging.info(f"initializer first 10 wights:{weights[:10]}")


def parser_tensor(tensor, use="normal"):
    name = tensor.name
    logging.info(f"{use} tensor name: {name}")

    data_type = tensor.type.tensor_type.elem_type
    logging.info(f"{use} tensor data type: {data_type}")

    dims = tensor.type.tensor_type.shape.dim
    shape = []
    for i, dim in enumerate(dims):
        shape.append(dim.dim_value)
    logging.info(f"{use} tensor with shape:{shape} ")


def parser_node(node):
    def attri_value(attri):
        if attri.type == 1:
            return attri.i
        elif attri.type == 7:
            return list(attri.ints)

    name = node.name
    logging.info(f"node name:{name}")

    opType = node.op_type
    logging.info(f"node op type:{opType}")

    inputs = list(node.input)
    logging.info(f"node with {len(inputs)} inputs:{inputs}")

    outputs = list(node.output)
    logging.info(f"node with {len(outputs)} outputs:{outputs}")

    attributes = node.attribute
    for attri in attributes:
        name = attri.name
        value = attri_value(attri)
        logging.info(f"{name} with value:{value}")


def parser_info(onnx_model):
    ir_version = onnx_model.ir_version
    producer_name = onnx_model.producer_name
    producer_version = onnx_model.producer_version
    for info in [ir_version, producer_name, producer_version]:
        logging.info("onnx model with info:{}".format(info))


def parser_inputs(onnx_graph):
    inputs = onnx_graph.input
    for input in inputs:
        parser_tensor(input, "input")


def parser_outputs(onnx_graph):
    outputs = onnx_graph.output
    for output in outputs:
        parser_tensor(output, "output")


def parser_graph_initializers(onnx_graph):
    initializers = onnx_graph.initializer
    for initializer in initializers:
        parser_initializer(initializer)


def parser_graph_nodes(onnx_graph):
    nodes = onnx_graph.node
    for node in nodes:
        parser_node(node)
        t = 1


def onnx_parser():
    model_path = "models/M208_26e_body.onnx"
    model = onnx.load(model_path)

    # 0.
    parser_info(model)

    graph = model.graph

    # 1.
    # parser_inputs(graph)

    # # 2.
    # parser_outputs(graph)

    # # 3.
    # parser_graph_initializers(graph)

    # # 4.
    parser_graph_nodes(graph)


# if __name__ == '__main__':
#     onnx_parser()

if __name__ == "__main__":
    model_path = "models/M208_26e_body.onnx"
    model = onnx.load(model_path)

    nodes = model.graph.node
    nodnum = len(nodes)  # 205
    for nid in range(nodnum):
        if nodes[nid].output[0] == "stride_32":
            print("Found stride_32: index = ", nid)
        else:
            print("input: ", nodes[nid].input, " output: ", nodes[nid].output)

    inits = model.graph.initializer
    ininum = len(inits)  # 124

    for iid in range(ininum):
        el = inits[iid]
        print(
            "name:", el.name, " dtype:", el.data_type, " dim:", el.dims
        )  # el.raw_data for weights and biases

    print(model.graph.output)  # display all the output nodes
    # print(model.graph.input)
