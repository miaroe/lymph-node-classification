import visualkeras
from PIL import ImageFont
import os


def visual_model(model, reports_path, model_arch):
    fig_name = model_arch + '_visual_model.png'
    #font = ImageFont.truetype("arial.ttf", 18)
    visualkeras.layered_view(model, legend=True,
                             to_file= os.path.join(reports_path, 'figures/') + fig_name)