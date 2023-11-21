# from torchview import draw_graph
from torchviz import make_dot
from configs import *
import os
# import graphviz

# when running on VSCode run the below command
# svg format on vscode does not give desired result
# graphviz.set_jupyter_format("png")

model = EfficientNetB3WithNorm(num_classes=7)

batch_size = 2
# device='meta' -> no memory is consumed for visualization
# model_graph = draw_graph(model, input_size=(32, 3, 224, 224), save_graph=True, filename="model_graph.png")
# model_graph.visual_graph

model_graph = make_dot(
    model(torch.randn(batch_size, 3, 224, 224)), params=dict(model.named_parameters())
).render("torchviz", format="png")
