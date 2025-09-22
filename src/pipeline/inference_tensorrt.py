
# pipeline/inference_tensorrt.py
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # ensures CUDA context stays alive
import tensorrt as trt
import torch

class InferenceTensorRT:
    def __init__(self, engine_path, max_batch=None, seq_len=None):
        self.max_batch = max_batch
        self.seq_len = seq_len

        if engine_path is None:
            raise ValueError("engine_path must be provided")

        # Load TensorRT engine
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load engine from: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Identify input/output tensor names
        self.input_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]

        if len(self.output_names) != 1:
            raise ValueError(f"Only single-output engines are supported, found {len(self.output_names)} outputs")

        self.output_name = self.output_names[0]
        print(f"✅ TensorRT engine loaded. Inputs: {self.input_names}, Output: {self.output_name}")

    def run(self, input_ids, attention_mask):
        if torch.is_tensor(input_ids):
            input_ids = input_ids.cpu().numpy()
        if torch.is_tensor(attention_mask):
            attention_mask = attention_mask.cpu().numpy()

        self.context.set_input_shape(self.input_names[0], input_ids.shape)
        self.context.set_input_shape(self.input_names[1], attention_mask.shape)

        # Allocate device memory
        d_input_ids = cuda.mem_alloc(input_ids.nbytes)
        d_attention_mask = cuda.mem_alloc(attention_mask.nbytes)

        output_shape = (input_ids.shape[0], self.context.get_tensor_shape(self.output_name)[-1])
        h_output = np.empty(output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(h_output.nbytes)

        # Transfer inputs
        cuda.memcpy_htod(d_input_ids, input_ids)
        cuda.memcpy_htod(d_attention_mask, attention_mask)

        # Bind device memory addresses
        bindings = [int(d_input_ids), int(d_attention_mask), int(d_output)]

        # Execute inference
        self.context.execute_v2(bindings)

        # Fetch output
        cuda.memcpy_dtoh(h_output, d_output)

        # Postprocess
        if h_output.shape[1] == 1:
            probs = 1 / (1 + np.exp(-h_output))
            return probs.flatten().tolist()
        else:
            probs = np.exp(h_output) / np.exp(h_output).sum(-1, keepdims=True)
            return probs[:, 1].tolist()

    def run_batch(self, input_ids_batch, attention_mask_batch):
        results = []
        for i in range(len(input_ids_batch)):
            res = self.run(input_ids_batch[i].unsqueeze(0), attention_mask_batch[i].unsqueeze(0))
            results.append(res)
        return results

    def cleanup(self):
        try:
            del self.engine, self.context
            torch.cuda.empty_cache()
            print("🧹 TensorRT resources cleaned up")
        except Exception as e:
            print(f"⚠️ Cleanup issue: {e}")
