import os
import tensorrt as trt

def build_or_load_engine(onnx_path: str, engine_path: str, fp16: bool = True):
    #return None
    """
    If engine exists, load it. Otherwise, build from ONNX.
    """

    if os.path.exists(engine_path):
        print(f"[INFO] Loading TensorRT engine from {engine_path}")
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine
    else:
        print(f"[INFO] Building TensorRT engine from ONNX: {onnx_path}")
        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))

        with open(onnx_path, "rb") as model:
            parser.parse(model.read())

        config = builder.create_builder_config()
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        engine = builder.build_engine(network, config)

        # Save engine
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        return engine
