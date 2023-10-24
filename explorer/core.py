from typing import Callable, List, Dict, Union, Optional, Tuple, Sequence, TypeVar
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, GraphProto, ModelProto, NodeProto
from typing import Dict, Optional, Union, cast, Text, IO
from google import protobuf
from colorama import init, Fore, Style, Back
import os
import sys
import numpy as np
from tabulate import tabulate
from rich.console import Console
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich import box
from .common import elem_type_to_name

def load_onnx_model(f, proto):
    s = None
    if hasattr(f, 'read') and callable(cast(IO[bytes], f).read):
        s = cast(IO[bytes], f).read()
    else:
        with open(cast(Text, f), 'rb') as readable:
            s = readable.read()
    decoded = cast(Optional[int], proto.ParseFromString(s))
    if decoded is not None and decoded != len(s):
        raise protobuf.message.DecodeError(
            "Protobuf decoding consumed too few bytes: {} out of {}".format(
                decoded, len(s)))
    return proto


def search_nodes_by_type(model, type_name):
    """
    search all nodes that with Type
    :param model_proto:
    :param type_name:
    :return:
    """
    print(Style.BRIGHT + 'search node by op type: ' +
          Style.RESET_ALL + Fore.GREEN + type_name + Style.RESET_ALL)
    if isinstance(model, ModelProto):
        graph = model.graph
        i = 0
        for node in graph.node:
            if type_name in node.op_type:
                i += 1
                print(node)
        print(Style.BRIGHT + 'listed all {} {} nodes in detail.\n'.format(i,
                                                                          type_name) + Style.RESET_ALL)


def list_nodes(model):
    """
    print out all nodes
    :param model_proto:
    :return:
    """
    print(Style.BRIGHT + 'start list nodes names:' + Style.RESET_ALL)
    if isinstance(model, ModelProto):
        graph = model.graph
        i = 0
        for node in graph.node:
            i += 1
            print('{}. {}'.format(i, node.name))
        print(Style.BRIGHT + 'listed all {} nodes names.\n'.format(i) + Style.RESET_ALL)


def list_nodes_hl(model):
    print(Style.BRIGHT + 'start list nodes all:' + Style.RESET_ALL)
    if isinstance(model, ModelProto):
        graph = model.graph
        i = 0
        for node in graph.node:
            if len(node.attribute) > 0 and node.attribute[0].t.raw_data != None:
                data_type = node.attribute[0].t.data_type
                if data_type == 7:
                    raw_data = np.frombuffer(
                        node.attribute[0].t.raw_data, dtype=np.int64)
                elif data_type == 1:
                    raw_data = np.frombuffer(
                        node.attribute[0].t.raw_data, dtype=np.float32)
                else:
                    raw_data = np.frombuffer(
                        node.attribute[0].t.raw_data, dtype=np.int64)
                print('|raw data that encoded inside this node: {}|'.format(raw_data))
            print(node)
            i += 1
        print(Style.BRIGHT +
              'listed all {} nodes in detail.\n'.format(i) + Style.RESET_ALL)


def search_node_by_id(model, node_id):
    """
    print out specific node by ID
    :param model_proto:
    :param node_id:
    :return:
    """
    print(Style.BRIGHT + 'search node by ID: ' +
          Style.RESET_ALL + Fore.GREEN + node_id + Style.RESET_ALL)
    if isinstance(model, ModelProto):
        graph = model.graph
        i = 0
        for node in graph.node:
            if node_id in node.name:
                i += 1
                print(node)
        print(Style.BRIGHT +
              'listed all {} nodes in detail.\n'.format(i) + Style.RESET_ALL)


def get_all_used_op_types(model):
    if isinstance(model, ModelProto):
        graph = model.graph
        all_op_types = {}
        for node in graph.node:
            if node.op_type not in all_op_types.keys():
                all_op_types[node.op_type] = 1
            else:
                all_op_types[node.op_type] += 1
    return all_op_types


def get_input_and_output_shapes(model, verbose=False) -> Union[Dict[str, list], None]:
    input_specs = dict()
    output_specs = dict()

    _inputs = get_inputs(model)
    for _input in _inputs:
        tensor_type = _input.type.tensor_type
        sh = []
        # check if it has a shape:
        if (tensor_type.HasField("shape")):
            # iterate through dimensions of the shape:
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if (d.HasField("dim_value")):
                    sh.append(d.dim_value)
                    if verbose:
                        print(d.dim_value, end=", ")  # known dimension
                elif (d.HasField("dim_param")):
                    sh.append(-1)
                    # unknown dimension with symbolic name
                    if verbose:
                        print(d.dim_param, end=", ")
                else:
                    sh.append('?')
                    if verbose:
                        print("?", end=", ")  # unknown dimension with no name
            input_specs[_input.name] = (sh, elem_type_to_name(tensor_type.elem_type))
        else:
            print("unknown rank", end="")

    for output in model.graph.output:
        if verbose:
            print(output.name, end=": ")
        # get type of input tensor
        tensor_type = output.type.tensor_type
        sh = []
        # check if it has a shape:
        if (tensor_type.HasField("shape")):
            # iterate through dimensions of the shape:
            for d in tensor_type.shape.dim:
                # the dimension may have a definite (integer) value or a symbolic identifier or neither:
                if (d.HasField("dim_value")):
                    sh.append(d.dim_value)
                    if verbose:
                        print(d.dim_value, end=", ")  # known dimension
                elif (d.HasField("dim_param")):
                    sh.append(-1)
                    # unknown dimension with symbolic name
                    if verbose:
                        print(d.dim_param, end=", ")
                else:
                    sh.append('?')
                    if verbose:
                        print("?", end=", ")  # unknown dimension with no name
            output_specs[output.name] = (sh, elem_type_to_name(tensor_type.elem_type))
        else:
            print("unknown rank", end="")

    return input_specs, output_specs


def summary(model_proto, model_p, model_name, verbose=False):
    if isinstance(model_proto, ModelProto):
        all_ops = get_all_used_op_types(model_proto)
        opi_vd = ''
        for opi in model_proto.opset_import:
            opi_vd += str(opi.version)
            opi_vd += opi.domain
            opi_vd += ', '
        summary_str = f'IR Version: {model_proto.ir_version}\n' + \
            f'Opset Version: {opi_vd}\n' + \
            f'Doc: {model_proto.doc_string}\n' + \
            f'Producer Name: {model_proto.producer_name}\n' + \
            f"All Ops: {','.join(all_ops.keys())}"

        console = Console()
        p = Panel(summary_str,
                  title=f'{model_name} Summary', highlight=True)
        console.print(p, width=90)

        inp_specs, oup_specs = get_input_and_output_shapes(
            model_proto, verbose=verbose)
        inp_specs = {k: (v[0], 'input', v[1]) for k, v in inp_specs.items()}
        oup_specs = {k: (v[0], 'output', v[1]) for k, v in oup_specs.items()}
        inp_specs.update(oup_specs)

        # print table

        table = Table(title=f'{model_name} Detail',
            caption='Table generated by modelexplorer', expand=True,
            row_styles=["none", "none"],
            pad_edge=True,
            box=box.ROUNDED,)
        table.add_column('[yellow]Name')
        table.add_column('[cyan]Shape')
        table.add_column('[blue]Input/Output')
        table.add_column('[green]Dtype')
        for k, v in inp_specs.items():
            table.add_row(k, str(v[0]), v[1], v[2])
        # panel_table = Panel(table, title=f'{model_name} Detail',)
        # console.print(panel_table)
        console.print(table, width=90)

        if verbose:
            # using table to show
            stat_data = []
            a_nodes = 0
            for k, v in all_ops.items():
                stat_data.append([k, v])
                a_nodes += v
            print(Fore.BLUE + Style.BRIGHT +
                  '\nStatistic table: ' + Style.RESET_ALL)
            print(tabulate(stat_data, headers=[
                'Op Name', 'Number']))
            print('Total {} kinds ops, {} nodes.'.format(len(stat_data), a_nodes))


def get_inputs(model: onnx.ModelProto) -> List[onnx.ValueInfoProto]:
    initializer_names = [x.name for x in model.graph.initializer]
    return [ipt for ipt in model.graph.input if ipt.name not in initializer_names]


def get_input_names(model: onnx.ModelProto) -> List[str]:
    input_names = [ipt.name for ipt in get_inputs(model)]
    return input_names
