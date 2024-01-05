import os
import pathlib

import hydra
import numpy as np
import onnx
import torch
from omegaconf import DictConfig
from onnxsim import simplify
from torch.nn import Identity

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.environment.cfg_utils import load_experiment_cfg
from super_gradients.common.environment.checkpoints_dir_utils import get_checkpoints_dir_path, get_latest_run_id
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.training import models
from super_gradients.training.utils.sg_trainer_utils import parse_args
from super_gradients.training.utils.utils import get_param

logger = get_logger(__name__)

ct = None

try:
    import coremltools as coreml_tools

    ct = coreml_tools
except (ImportError, ModuleNotFoundError):
    pass


class ConvertableCompletePipelineModel(torch.nn.Module):
    """
    Exportable nn.Module that wraps the model, preprocessing and postprocessing.

    :param model: torch.nn.Module, the main model. takes input from pre_process' output, and feeds pre_process.
    :param pre_process: torch.nn.Module, preprocessing module, its output will be model's input. When none (default), set to Identity().
    :param pre_process: torch.nn.Module, postprocessing module, its output is the final output. When none (default), set to Identity().
    :param **prep_model_for_conversion_kwargs: for SgModules- args to be passed to model.prep_model_for_conversion
            prior to torch.onnx.export call.
    """

    def __init__(self, model: torch.nn.Module, pre_process: torch.nn.Module = None, post_process: torch.nn.Module = None, **prep_model_for_conversion_kwargs):
        super(ConvertableCompletePipelineModel, self).__init__()
        model.eval()
        pre_process = pre_process or Identity()
        post_process = post_process or Identity()
        if hasattr(model, "prep_model_for_conversion"):
            model.prep_model_for_conversion(**prep_model_for_conversion_kwargs)
        self.model = model
        self.pre_process = pre_process
        self.post_process = post_process

    def forward(self, x):
        return self.post_process(self.model(self.pre_process(x)))


def pipeline_coreml(model, weights_dir=None):
    import coremltools as ct

    _, _, h, w = 0, 0, 640, 640

    # Output shapes
    spec = model.get_spec()
    print(spec.description.output)
    out0, out1 = iter(spec.description.output)

    from PIL import Image
    img = Image.new('RGB', (w, h))  # w=192, h=320
    out = model.predict({'x_1': img})
    out0_shape = out[out0.name].shape  # (8400, 4)
    out1_shape = out[out1.name].shape  # (8400, 2)

    # Checks
    names = ["hoop", "score"]
    nx, ny = spec.description.input[0].type.imageType.width, spec.description.input[0].type.imageType.height
    nc = 2
    assert len(names) == nc, f'{len(names)} names found for nc={nc}'  # check

    # Define output shapes (missing)
    out0.type.multiArrayType.shape[:] = out0_shape  # (8400, 4)
    out1.type.multiArrayType.shape[:] = out1_shape  # (8400, 2)

    # Model from spec
    model = ct.models.MLModel(spec, weights_dir=weights_dir)

    # 3. Create NMS protobuf
    nms_spec = ct.proto.Model_pb2.Model()
    nms_spec.specificationVersion = 5
    for i in range(2):
        decoder_output = model._spec.description.output[i].SerializeToString()
        nms_spec.description.input.add()
        nms_spec.description.input[i].ParseFromString(decoder_output)
        nms_spec.description.output.add()
        nms_spec.description.output[i].ParseFromString(decoder_output)

    nms_spec.description.output[0].name = 'confidence'
    nms_spec.description.output[1].name = 'coordinates'

    output_sizes = [nc, 4]
    for i in range(2):
        ma_type = nms_spec.description.output[i].type.multiArrayType
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[0].lowerBound = 0
        ma_type.shapeRange.sizeRanges[0].upperBound = -1
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
        ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
        del ma_type.shape[:]

    nms = nms_spec.nonMaximumSuppression
    nms.confidenceInputFeatureName = out1.name  # 1x507x80
    nms.coordinatesInputFeatureName = out0.name  # 1x507x4
    nms.confidenceOutputFeatureName = 'confidence'
    nms.coordinatesOutputFeatureName = 'coordinates'
    nms.iouThresholdInputFeatureName = 'iouThreshold'
    nms.confidenceThresholdInputFeatureName = 'confidenceThreshold'
    nms.iouThreshold = 0.45
    nms.confidenceThreshold = 0.25
    nms.pickTop.perClass = False
    nms.stringClassLabels.vector.extend(names)
    nms_model = ct.models.MLModel(nms_spec)

    # 4. Pipeline models together
    pipeline = ct.models.pipeline.Pipeline(input_features=[('x_1', ct.models.datatypes.Array(3, ny, nx)),
                                                            ('iouThreshold', ct.models.datatypes.Double()),
                                                            ('confidenceThreshold', ct.models.datatypes.Double())],
                                            output_features=['confidence', 'coordinates'])

    pipeline.add_model(model)
    pipeline.add_model(nms_model)

    # Correct datatypes
    pipeline.spec.description.input[0].ParseFromString(model._spec.description.input[0].SerializeToString())
    pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
    pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

    # Update metadata
    pipeline.spec.specificationVersion = 5
    pipeline.spec.description.metadata.userDefined.update({
        'IoU threshold': str(nms.iouThreshold),
        'Confidence threshold': str(nms.confidenceThreshold)})

    # Save the model
    model = ct.models.MLModel(pipeline.spec, weights_dir=weights_dir)

    # model.input_description['image'] = 'Input image'
    model.input_description['iouThreshold'] = f'(optional) IOU threshold override (default: {nms.iouThreshold})'
    model.input_description['confidenceThreshold'] = \
        f'(optional) Confidence threshold override (default: {nms.confidenceThreshold})'
    model.output_description['confidence'] = 'Boxes × Class confidence (see user-defined metadata "classes")'
    model.output_description['coordinates'] = 'Boxes × [x, y, width, height] (relative to image size)'
    return model

@resolve_param("pre_process", TransformsFactory())
@resolve_param("post_process", TransformsFactory())
def convert_to_coreml(
    model: torch.nn.Module,
    out_path: str,
    input_size: tuple = None,
    pre_process: torch.nn.Module = None,
    post_process: torch.nn.Module = None,
    prep_model_for_conversion_kwargs=None,
    export_as_ml_program=False,
    torch_trace_kwargs=None,
):
    """
        Exports a given SG model to CoreML mlprogram or package.

        :param model: torch.nn.Module, model to export to CoreML.
        :param out_path: str, destination path for the .mlmodel file.
        :param input_size: Input shape without batch dimensions ([C,H,W]). Batch size assumed to be 1.
        :param pre_process: torch.nn.Module, preprocessing pipeline, will be resolved by TransformsFactory()
        :param post_process: torch.nn.Module, postprocessing pipeline, will be resolved by TransformsFactory()
        :param prep_model_for_conversion_kwargs: dict, for SgModules- args to be passed to model.prep_model_for_conversion
         prior to ct.convert call. Supported keys are:
        - input_size - Shape of inputs with batch dimension, [C,H,W] for image inputs.
        :param export_as_ml_program: Whether to convert to the new program format (better) or legacy coreml proto file
                            (Supports more iOS versions and devices, but this format will be deprecated at some point).
        :param torch_trace_kwargs: kwargs for torch.jit.trace
    :return: Path
    """
    if ct is None:
        raise ImportError(
            '"coremltools" is required for CoreML export, but is not installed. Please install CoreML Tools using:\n'
            '   "python3 -m pip install coremltools" and try again (Tested with version 6.3.0);'
        )

    logger.debug("Building model...")
    logger.debug(model)
    logger.debug("Model child nodes:")
    logger.debug(next(model.named_children()))

    if not os.path.isdir(pathlib.Path(out_path).parent.resolve()):
        raise FileNotFoundError(f"Could not find destination directory {out_path} for the CoreML file.")
    torch_trace_kwargs = torch_trace_kwargs or dict()
    prep_model_for_conversion_kwargs = prep_model_for_conversion_kwargs or dict()

    if input_size is not None:
        input_size = (1, *input_size)
        logger.warning(
            f"input_shape is deprecated and will be removed in the next major release."
            f"Use the convert_to_coreml(..., prep_model_for_conversion_kwargs(input_size={input_size})) instead"
        )
        prep_model_for_conversion_kwargs["input_size"] = input_size

    if "input_size" not in prep_model_for_conversion_kwargs:
        raise KeyError("input_size must be provided in prep_model_for_conversion_kwargs")

    input_size = prep_model_for_conversion_kwargs["input_size"]

    # TODO: support more than 1 input when prep_for_conversoin will support it.
    example_inputs = [torch.Tensor(np.zeros(input_size))]

    if not out_path.endswith(".mlpackage") and not out_path.endswith(".mlmodel"):
        out_path += ".mlpackage" if export_as_ml_program else ".mlmodel"

    complete_model = ConvertableCompletePipelineModel(model, pre_process, post_process, **prep_model_for_conversion_kwargs)

    # Set the model in evaluation mode.
    complete_model.eval()

    logger.info("Creating torch jit trace...")
    traced_model = torch.jit.trace(complete_model, example_inputs, **torch_trace_kwargs)
    logger.info("Tracing the model with the provided inputs...")
    out = traced_model(*example_inputs)  # using * because example_inputs is a list
    # logger.info(f"Inferred output shapes: {[o.shape for o in out]}")
    if export_as_ml_program:
        coreml_model = ct.convert(
            traced_model, convert_to="mlprogram", inputs=[ct.ImageType(name=f"x_{i + 1}", shape=_.shape) for i, _ in enumerate(example_inputs)]
        )
    else:
        coreml_model = ct.convert(traced_model, inputs=[ct.ImageType(name=f"x_{i + 1}", shape=_.shape) for i, _ in enumerate(example_inputs)])

    coreml_model = pipeline_coreml(coreml_model, weights_dir=coreml_model.weights_dir)

    spec = coreml_model.get_spec()
    logger.debug(spec.description)

    # Changing the input names:
    #   In CoreML, the input name is compiled into classes (named keyword argument in predict).
    #   We want to re-use the same names among different models to make research easier.
    #   We normalize the inputs names to be x_1, x_2, etc.
    for i, _input in enumerate(spec.description.input):
        new_input_name = "x_" + str(i + 1)
        logger.info(f"Renaming input {_input.name} to {new_input_name}")
        ct.utils.rename_feature(spec, _input.name, new_input_name)

    # Re-Initializing the model with the new spec
    coreml_model = ct.models.MLModel(spec, weights_dir=coreml_model.weights_dir)

    # Saving the model
    coreml_model.save(out_path)
    logger.info(f"CoreML model successfully save to {os.path.abspath(out_path)}")
    return out_path


@resolve_param("pre_process", TransformsFactory())
@resolve_param("post_process", TransformsFactory())
def convert_to_onnx(
    model: torch.nn.Module,
    out_path: str,
    input_shape: tuple = None,
    pre_process: torch.nn.Module = None,
    post_process: torch.nn.Module = None,
    prep_model_for_conversion_kwargs=None,
    torch_onnx_export_kwargs=None,
    simplify: bool = True,
):
    """
    Exports model to ONNX.

    :param model: torch.nn.Module, model to export to ONNX.
    :param out_path: str, destination path for the .onnx file.
    :param input_shape: Input shape without batch dimensions ([C,H,W]). Batch size assumed to be 1.
    DEPRECATED USE input_size KWARG IN prep_model_for_conversion_kwargs INSTEAD.
    :param pre_process: torch.nn.Module, preprocessing pipeline, will be resolved by TransformsFactory()
    :param post_process: torch.nn.Module, postprocessing pipeline, will be resolved by TransformsFactory()
    :param prep_model_for_conversion_kwargs: dict, for SgModules- args to be passed to model.prep_model_for_conversion
     prior to torch.onnx.export call. Supported keys are:
    - input_size - Shape of inputs with batch dimension, [C,H,W] for image inputs.
    :param torch_onnx_export_kwargs: kwargs (EXCLUDING: FIRST 3 KWARGS- MODEL, F, ARGS). to be unpacked in torch.onnx.export call
    :param simplify: bool,whether to apply onnx simplifier method, same as `python -m onnxsim onnx_path onnx_sim_path.
     When true, the simplified model will be saved in out_path (default=True).

    :return: out_path
    """
    if not os.path.isdir(pathlib.Path(out_path).parent.resolve()):
        raise FileNotFoundError(f"Could not find destination directory {out_path} for the ONNX file.")
    torch_onnx_export_kwargs = torch_onnx_export_kwargs or dict()
    prep_model_for_conversion_kwargs = prep_model_for_conversion_kwargs or dict()

    if input_shape is not None:
        input_size = (1, *input_shape)
        logger.warning(
            f"input_shape is deprecated and will be removed in the next major release."
            f"Use the convert_to_onnx(..., prep_model_for_conversion_kwargs(input_size={input_size})) instead"
        )
        prep_model_for_conversion_kwargs["input_size"] = input_size

    if "input_size" not in prep_model_for_conversion_kwargs:
        raise KeyError("input_size must be provided in prep_model_for_conversion_kwargs")

    input_size = prep_model_for_conversion_kwargs["input_size"]

    onnx_input = torch.Tensor(np.zeros(input_size))
    if not out_path.endswith(".onnx"):
        out_path = out_path + ".onnx"
    complete_model = ConvertableCompletePipelineModel(model, pre_process, post_process, **prep_model_for_conversion_kwargs)

    torch.onnx.export(model=complete_model, args=onnx_input, f=out_path, **torch_onnx_export_kwargs)
    if simplify:
        onnx_simplify(out_path, out_path)
    return out_path


def prepare_conversion_cfgs(cfg: DictConfig):
    """
    Builds the cfg (i.e conversion_params) and experiment_cfg (i.e recipe config according to cfg.experiment_name)
     to be used by convert_recipe_example

    :param cfg: DictConfig, converion_params config
    :return: cfg, experiment_cfg
    """
    cfg = hydra.utils.instantiate(cfg)
    # CREATE THE EXPERIMENT CFG

    # Load the latest experiment config
    run_id = get_param(cfg, "run_id")
    if run_id is None:
        run_id = get_latest_run_id(experiment_name=cfg.experiment_name, checkpoints_root_dir=cfg.ckpt_root_dir)
    experiment_cfg = load_experiment_cfg(ckpt_root_dir=cfg.ckpt_root_dir, experiment_name=cfg.experiment_name, run_id=run_id)

    hydra.utils.instantiate(experiment_cfg)
    if cfg.checkpoint_path is None:
        logger.info(
            "checkpoint_params.checkpoint_path was not provided, so the model will be converted using weights from "
            "checkpoints_dir/training_hyperparams.ckpt_name "
        )
        checkpoints_dir = get_checkpoints_dir_path(experiment_name=cfg.experiment_name, ckpt_root_dir=cfg.ckpt_root_dir, run_id=run_id)
        cfg.checkpoint_path = os.path.join(checkpoints_dir, cfg.ckpt_name)

    cfg.out_path = cfg.out_path or cfg.checkpoint_path.replace(".pth", ".onnx")
    logger.info(f"Exporting checkpoint: {cfg.checkpoint_path} to ONNX.")
    return cfg, experiment_cfg


def convert_from_config(cfg: DictConfig) -> str:
    """
    Exports model according to cfg.

    See:
     super_gradients/recipes/conversion_params/default_conversion_params.yaml for the full cfg content documentation,
     and super_gradients/examples/convert_recipe_example/convert_recipe_example.py for usage.
    :param cfg:
    :return: out_path, the path of the saved .onnx file.
    """
    cfg, experiment_cfg = prepare_conversion_cfgs(cfg)
    model = models.get(
        model_name=experiment_cfg.architecture,
        num_classes=experiment_cfg.arch_params.num_classes,
        arch_params=experiment_cfg.arch_params,
        strict_load=cfg.strict_load,
        checkpoint_path=cfg.checkpoint_path,
    )
    cfg = parse_args(cfg, models.convert_to_onnx)
    out_path = models.convert_to_onnx(model=model, **cfg)
    logger.info(f"Successfully exported model at {out_path}")
    return out_path


def onnx_simplify(onnx_path: str, onnx_sim_path: str):
    """
    onnx simplifier method, same as `python -m onnxsim onnx_path onnx_sim_path
    :param onnx_path: path to onnx model
    :param onnx_sim_path: path for output onnx simplified model
    """
    model_sim, check = simplify(model=onnx_path)
    if not check:
        raise RuntimeError("Simplified ONNX model could not be validated")
    onnx.save_model(model_sim, onnx_sim_path)
