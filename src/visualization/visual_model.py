import visualkeras
from PIL import ImageFont
import os


def visual_model(model, reports_path, model_arch):
    fig_name = model_arch + '_visual_model.png'
    fig_path = os.path.join(reports_path, 'figures/')
    os.makedirs(fig_path, exist_ok=True)
    #font = ImageFont.truetype("arial.ttf", 18)
    visualkeras.layered_view(model, legend=True,
                             to_file=fig_path + fig_name)