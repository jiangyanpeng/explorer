from alfred.deploy.tensorrt.common import (
    build_engine_onnx_v2,
    check_engine,
)
from alfred.utils.log import logger
import os


def convert_onnx_to_tensorrt(onnx_f, datatype, batch_size=1, min_shapes=None, opt_shapes=None, max_shapes=None, with_suffix=True):
    opt_params = None
    if min_shapes is not None and opt_shapes is not None and max_shapes is not None:
        names = [i.split(':')[0] for i in min_shapes]
        mins_shapes = [[int(ii) for ii in i.split(':')[1].split('x')]
                       for i in min_shapes]
        opt_shapes = [[int(ii) for ii in i.split(':')[1].split('x')]
                      for i in opt_shapes]
        max_shapes = [[int(ii) for ii in i.split(':')[1].split('x')]
                      for i in max_shapes]

        opt_params = dict()
        for i in range(len(min_shapes)):
            opt_params[names[i]] = [
                mins_shapes[i], opt_shapes[i], max_shapes[i]]
        batch_size = '_dynamic_' + str(max_shapes[0][0])

    fp16 = False
    int8 = False
    if datatype == 16:
        logger.info('enable fp16 mode.')
        fp16 = True
    elif datatype == 8:
        print("int8 not supported.")
    if with_suffix:
        if fp16:
            save_p = os.path.join(os.path.dirname(onnx_f), os.path.basename(
                onnx_f).replace(".onnx", "_fp16_bs{}.engine".format(batch_size)))
        elif int8:
            save_p = os.path.join(os.path.dirname(onnx_f), os.path.basename(
                onnx_f).replace(".onnx", "_int8_bs{}.engine".format(batch_size)))
        else:
            save_p = os.path.join(os.path.dirname(onnx_f), os.path.basename(
                onnx_f).replace(".onnx", "_bs{}.engine".format(batch_size)))
    else:
        save_p = os.path.join(os.path.dirname(onnx_f), os.path.basename(
            onnx_f).replace(".onnx", ".engine"))
    logger.info('engine will be saved into: {}'.format(save_p))
    engine = build_engine_onnx_v2(
        onnx_file_path=onnx_f, max_batch_size=batch_size, engine_file_path=save_p, fp16_mode=fp16, save_engine=True, opt_params=opt_params
    )
    # check_engine(engine)
    logger.info('engine saved into: {}'.format(save_p))
