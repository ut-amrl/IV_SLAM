import sys, os, argparse
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

from skimage import io
import torch
import torchvision
from torchvision import transforms
import numpy as np
import cv2

from config import cfg
from networks.models_light import ModelBuilder, IntrospectionModule
from lib.utils.utils import MaskedMSELoss

# This script converts the simplified version of the pytorch model to a torch 
# script via tracing or annotation

# Convert using tracing or annotation
USE_TRACING=True
LOAD_EXAMPLE_INPUT_FROM_FILE=False

def main():
  parser = argparse.ArgumentParser(description='Convert Pytorch models '
                                    'to Torch Script for use in CPP.')
  parser.add_argument(
                      "--cfg",
                      default=None,
                      metavar="FILE",
                      help="path to config file",
                      required=True,
                      type=str)
  parser.add_argument("--output_model",
                    default=None,
                    help="Path to the converted torch script model",
                    required=True)
  args = parser.parse_args()

  cfg.merge_from_file(args.cfg)

  cfg.MODEL.weights_encoder = cfg.TEST.test_model_encoder
  cfg.MODEL.weights_decoder = cfg.TEST.test_model_decoder

  apply_logistic_func = True

  # The desired size of the input image to the network. The model resizes the 
  # input image to this size before feeding it to the encoder
  desired_input_size = (cfg.DATASET.img_height,
                        cfg.DATASET.img_width)

  # The desired size of the output image. The model interpolates the output
  # to this size
  desired_output_size = (cfg.TEST.output_img_height,
                         cfg.TEST.output_img_width)

  # Build the network from selected modules
  net_encoder = ModelBuilder.build_encoder(
    arch=cfg.MODEL.arch_encoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    weights=cfg.MODEL.weights_encoder)
  net_decoder = ModelBuilder.build_decoder(
    arch=cfg.MODEL.arch_decoder.lower(),
    fc_dim=cfg.MODEL.fc_dim,
    num_class=cfg.DATASET.num_class,
    weights=cfg.MODEL.weights_decoder,
    regression_mode=cfg.MODEL.is_regression_mode,
    inference_mode=True,
    out_size=desired_output_size)

  if cfg.MODEL.is_regression_mode:
      if cfg.TRAIN.use_masked_loss:
        criterion = MaskedMSELoss()
        print("Regression Mode with Masked Loss")
      else:
        criterion = nn.MSELoss(reduction='mean')
        print("Regression Mode")
  else:
    criterion = nn.NLLLoss(ignore_index=-1)
    print("Segmentation Mode")

  net = IntrospectionModule(
        net_encoder, net_decoder,
        enc_input_size=desired_input_size,
        logistic_func=apply_logistic_func)

  # *******************
  # Example image input:
  if LOAD_EXAMPLE_INPUT_FROM_FILE:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    data_transform_input = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    example_img = "/media/ssd2/datasets/Jackal_Visual_Odom/sequences/00037/image_0/000984.png"

    img = io.imread(example_img)
    if len(img.shape) > 2:
      img = img[:, : , 0:3]
      img = img.reshape((img.shape[0], img.shape[1], 3))
    else:
      img = img.reshape((img.shape[0], img.shape[1], 1))
      img = np.repeat(img, 3, axis=2)
    

    img = transforms.ToPILImage()(img)
    img = data_transform_input(img)
    img = img.reshape((1, 3, img.shape[1], img.shape[2]))
  else:
    img = torch.rand(1, 3, 600, 960)
  # ***********************

  # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
  if USE_TRACING:
    net.eval()
    script_module = torch.jit.trace(net, img)
    script_module.save(args.output_model)
  else:
    # Convert the pytorch model to a ScriptModule via annotation
    script_module = torch.jit.script(net)
    script_module.save(args.output_model)

  

  # ************************************
  # Test with an example image
  # ************************************
  script_module.eval()
  net.eval()

  for model, name in zip([script_module, net], ['scrpt_model', 'orig_model']):
    
    print('input img: ', img.shape)
    with torch.set_grad_enabled(False):
      output_img = model(img)
    print(type(output_img))
    print("output_img: ", output_img.shape)


    # Visualize input img
    input_img_np = img.numpy()
    input_img_np = input_img_np[0, 0, :, :]
    # Unnormalize the R channel of the input image:
    input_img_np = (input_img_np * 0.229) + 0.485
    input_img_cv8u = (255.0 * input_img_np).astype(np.uint8)
    input_img_cv8uc3 = cv2.cvtColor(input_img_cv8u, cv2.COLOR_GRAY2BGRA)
    cv2.imwrite("example_input.png", input_img_cv8uc3)

    # ***************************
    # Visualize output img
    # ***************************
    output_np = output_img.numpy()
    output_np = output_np[0, 0, :, :]

    output_np = (255.0 * output_np).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(output_np, cv2.COLORMAP_JET)
    cv2.imwrite("output_" + name + ".png", heatmap_color)



if __name__=="__main__":
  main()