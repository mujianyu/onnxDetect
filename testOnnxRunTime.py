import onnxruntime as ort
 
# 创建一个推理会话
session = ort.InferenceSession(r"best.onnx", providers=['CUDAExecutionProvider'])
 
# 检查是否使用了CUDA
providers = session.get_providers()
print(f"Available providers: {providers}")
 
# 获取当前执行程序的是否使用GPU设备
device = ort.get_device()
print(f"Current device: {device}")