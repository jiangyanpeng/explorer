from explorer.explorer import Onnxexplorer
from explorer.model import Model
import numpy as np

if __name__ == "__main__":
    # Onnxexplorer()

    modelpath = 'models/M208_26e_body.onnx'
    m = Model(modelpath)
    # update tensor shapes with new input tensor
    m.graph.shape_infer({'data': np.zeros((1, 3, 224, 224))})
    m.graph.profile()
    # m.graph.print_node_map()  # console print
    m.graph.msg_console()
