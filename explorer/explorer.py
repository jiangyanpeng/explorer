from datetime import datetime
from .version import __version__, short_version
import os
import sys
import argparse
from tabnanny import verbose
import onnx
from onnx import ModelProto, TensorProto, NodeProto
from typing import Optional, cast, Text, IO
from google import protobuf
from colorama import init, Fore, Style, Back
from .core import *
from .msg import print_welcome_msg
from .model import Model

try:
    from .quant import convert_onnx_to_tensorrt
except ImportError:
    print("disable TensorRT since it was not found.")

try:
    from .quant import convert_onnx_to_qnn
except ImportError:
    print("disable QNN since it was not found.")

try:
    import onnxruntime as ort
except ImportError:
    print("explorer needs onnxruntime, you need install onnxruntime first.")
    exit(0)

init()
dt = datetime.now()


"""
This script provides
various exploration on onnx model
such as those operations:

- summary:
    this command will summarize model info mation such as:
    1. model opset version;
    2. if check pass;
    3. whether can be simplifiered or not;
    4. inputs node and outputs node;
    5. all nodes number;
    6. Initializers tensornumber and their index;

- search:
    1. search all nodes by OpType;
    2. search node by node name (ID);

- list:
    1. print out all nodes id
    2. list -hl will print all nodes and it's attribute

"""


def args_parse():
    """
    parse arguments
    divided into the following command line sections:
    common sections:
        --version : show explorer version
        --help : explorer user's guidance

    glance sections:
        --model : explorer input model path, now only support onnx model
        --verbose :

    profile sections:
        --model : explorer input model path

    optimizer sections:
        --model : explorer input model path

    modify sections:
        --model : explorer input model path

    check sections:
        --model : check onnx model is valid
        --print : print out all nodes id
        checktrt : check TensorRT env valid
        checkqnn : check QNN env valid

    quant sections:
        --model : quant model path
        --data_type : quantization bandwidth
    """
    parser = argparse.ArgumentParser(prog="explorer")
    parser.add_argument(
        "--version", "-v", action="store_true", help="show version info."
    )

    main_sub_parser = parser.add_subparsers(dest="subparser_name")

    # =============== glance part ================
    vision_parser = main_sub_parser.add_parser(
        "glance", help="take a glance at your onnx model."
    )
    vision_parser.add_argument("--model", "-m", help="onnx model path")
    vision_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
    )

    # =============== profiling part ================
    profiling = main_sub_parser.add_parser(
        "profile",
        help="profiling model info",
    )
    profiling.add_argument("--model", "-m", help="onnx model path")

    # =============== check part ================
    check_parser = main_sub_parser.add_parser(
        "check", help="Check your onnx model is valid or not."
    )
    check_parser.add_argument("--model", "-m", help="onnx model path")
    check_parser.add_argument("--print", action="store_true")

    checktrt_parser = main_sub_parser.add_parser(
        "checktrt", help="Check your trt model is valid or not."
    )
    checktrt_parser.add_argument("--model", "-m", help="onnx model path")

    # =============== quant part ================
    trt_parser = main_sub_parser.add_parser(
        "quant", help="Convert your onnx model to tensorrt engine."
    )
    trt_parser.add_argument("--model", "-m", help="onnx model path")
    trt_parser.add_argument(
        "--data_type",
        "-d",
        type=int,
        default=32,
        help="32, 16, 8 presents fp32, fp16, int8",
    )
    trt_parser.add_argument(
        "--batch_size", "-b", type=int, default=1, help="batch size of your model."
    )
    trt_parser.add_argument("--min_shapes", nargs="+", help="min_shapes of opt_params")
    trt_parser.add_argument("--opt_shapes", nargs="+", help="opt_shapes of opt_params")
    trt_parser.add_argument("--max_shapes", nargs="+", help="max_shapes of opt_params")
    trt_parser.add_argument(
        "--suffix",
        action="store_true",
        help="if enable, generated engine will have a suffix.",
    )

    return parser.parse_args()


class OnnxExplorer(object):
    def __init__(self):
        args = args_parse()
        if args.version:
            # show version and simply user's guidance
            print_welcome_msg(version=__version__)
        else:
            if args.subparser_name == None:
                print("should provide at least one sub command, -h for detail.")
                exit(-1)
            self.model_path = args.model
            if args.subparser_name == "checktrt":
                print(
                    Style.BRIGHT
                    + "Exploring on trt model: "
                    + Style.RESET_ALL
                    + Fore.GREEN
                    + self.model_path
                    + Style.RESET_ALL
                )
                self.check_trt_engine(self.model_path)
            elif args.model != None and os.path.exists(args.model):
                print(
                    Style.BRIGHT
                    + "Exploring on onnx model: "
                    + Style.RESET_ALL
                    + Fore.GREEN
                    + self.model_path
                    + Style.RESET_ALL
                )
                # self.model_proto = load_onnx_model(self.model_path, ModelProto())
                self.model_proto = onnx.load(self.model_path)
                if args.subparser_name == "glance":
                    summary(
                        self.model_proto,
                        self.model_path,
                        os.path.basename(args.model),
                        args.verbose,
                    )
                elif args.subparser_name == "quant":
                    convert_onnx_to_tensorrt(
                        self.model_path,
                        args.data_type,
                        args.batch_size,
                        args.min_shapes,
                        args.opt_shapes,
                        args.max_shapes,
                        with_suffix=args.suffix,
                    )
                elif args.subparser_name == "check":
                    self.check(args.print)
                elif args.subparser_name == "profile":
                    m = Model(self.model_path)
                    inp_specs, _ = get_input_and_output_shapes(self.model_proto)
                    m.graph.shape_infer(
                        {k: np.random.rand(*v[0]) for k, v in inp_specs.items()}
                    )
                    m.graph.profile()
                    m.graph.msg_console()

            else:
                print(
                    "{} does not exist or you should provide model path like `explorer glance -m model.onnx`.".format(
                        args.model
                    )
                )

    def ls(self):
        parser = argparse.ArgumentParser(description="ls at any level about the model.")
        # prefixing the argument with -- means it's optional
        parser.add_argument("-hl", action="store_true")
        args = parser.parse_args(sys.argv[3:])
        print("Running explorer ls, hl=%s" % args.hl)
        if args.hl:
            list_nodes_hl(self.model_proto)
        else:
            list_nodes(self.model_proto)

    def search(self):
        parser = argparse.ArgumentParser(
            description="search model node by name or type"
        )
        # NOT prefixing the argument with -- means it's not optional
        parser.add_argument("--name", "-n", help="name of to search node")
        parser.add_argument("--type", "-t", help="type of to search node")
        args = parser.parse_args(sys.argv[3:])
        if args.name != None:
            search_node_by_id(self.model_proto, args.name)
        elif args.type != None:
            search_nodes_by_type(self.model_proto, args.type)
        else:
            print("search should provide type name or id to search.")

    def check(self, p):
        print("checking on model: {}".format(self.model_path))
        onnx.checker.check_model(self.model_proto)
        if p:
            a = onnx.helper.printable_graph(self.model_proto.graph)
            print(a)

    # def check_trt_engine(self, trt_engine_f):
    #     from alfred.deploy.tensorrt.common import check_engine, load_engine_from_local

    #     if os.path.exists(trt_engine_f):
    #         engine = load_engine_from_local(trt_engine_f)
    #         check_engine(engine, do_print=True)
    #     else:
    #         print("can not found file: {}".format(trt_engine_f))


def main():
    OnnxExplorer()
