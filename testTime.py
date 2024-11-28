import onnxruntime as ort
import numpy as np
import time
 
# 生成随机输入数据（根据模型的输入要求调整大小）
input_shape = (1, 6, 640, 640)  # 假设模型需要 640x640 大小的图像输入
input_data = np.random.rand(*input_shape).astype(np.float32)
 
# 加载 ONNX 模型路径
model_path = r"best.onnx"
 
# 自定义运行次数，默认为 1000
def benchmark(session, input_data, iterations=1000):
    start_time = time.time()
    for _ in range(iterations):
        session.run(None, {session.get_inputs()[0].name: input_data})
    return time.time() - start_time
 
# 1. 使用 CPU 进行推理
session_cpu = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
cpu_iterations = 1000  # 可修改为自定义次数
cpu_time = benchmark(session_cpu, input_data, iterations=cpu_iterations)
print(f"CPU 推理总时间: {cpu_time:.4f} 秒, 每次推理平均时间: {cpu_time / cpu_iterations:.4f} 秒")
 
# 2. 使用 GPU (CUDA) 进行推理
session_gpu = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
gpu_iterations = 1000  # 可修改为自定义次数
gpu_time = benchmark(session_gpu, input_data, iterations=gpu_iterations)
print(f"GPU 推理总时间: {gpu_time:.4f} 秒, 每次推理平均时间: {gpu_time / gpu_iterations:.4f} 秒")
 
# 比较 GPU 和 CPU 的速度
speedup = cpu_time / gpu_time
print(f"GPU 加速比: {speedup:.2f} 倍")