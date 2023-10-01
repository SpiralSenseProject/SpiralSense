import torch
import onnx2tf
from configs import *

torch.onnx.export(model=MODEL, args=torch.randn(1, 3, 64, 64), f='output/checkpoints/model.onnx', verbose=True, input_names=['input'], output_names=['output'])

onnx2tf.convert(input_onnx_file_path='output/checkpoints/model.onnx', output_folder_path='output/checkpoints/converted/')