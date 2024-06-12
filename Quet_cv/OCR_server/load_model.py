from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from Quet_cv.OCR_server.craft import CRAFT
from Quet_cv.OCR_server.testCraft import copyStateDict
from Quet_cv.OCR_server.crop_box_img import YOLO_Detect
import argparse
import Quet_cv.OCR_server.pipeline as pipeline
import torch
import torch.backends.cudnn as cudnn

def get_model():
    # Detect 
    weight_path = r'G:\RecommendSysterm\Quet_cv\OCR_server\weights\best.pt'
    det_box_model = YOLO_Detect(weight_path=weight_path)
    # Recognition 
    config = Cfg.load_config_from_file(r'G:\RecommendSysterm\Quet_cv\OCR_server\configOCR\vgg-seq2seq.yml')
    config['weights'] = r'G:\RecommendSysterm\Quet_cv\OCR_server\weights\vgg_seq2seq.pth'
    config['export'] = 'vgg_seq2seq_checkpoint.pth'
    config['device'] = 'cpu'
    config['predictor']['beamsearch'] = False
    ocr_model = Predictor(config)
    # Recognition Info, name, Job
    config = Cfg.load_config_from_file(r'G:\RecommendSysterm\Quet_cv\OCR_server\configOCR\vgg-transformer.yml')
    config['weights'] = r'G:\RecommendSysterm\Quet_cv\OCR_server\weights\transformerocr.pth'
    config['export'] = 'vgg_transformer_checkpoint.pth'
    config['device'] = 'cpu'
    
    ocr_model_1 = Predictor(config)
    
    
    # CRAFT 
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default=r'G:\RecommendSysterm\Quet_cv\OCR_server\weights\craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cpu', default=True, type=pipeline.str2bool, help='Use cpu for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--refine', default=True, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default=r'G:\RecommendSysterm\Quet_cv\OCR_server\weights\craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

    args = parser.parse_args()
    craft = CRAFT()

    craft.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))
    if args.cpu:
        craft = craft.cpu()
        craft = torch.nn.DataParallel(craft)
        cudnn.benchmark = False
    refine_net = None
    if args.refine:
        from Quet_cv.OCR_server.refinenet import RefineNet
        refine_net = RefineNet()
        if args.cpu:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))
            refine_net = refine_net.cpu()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True
        
    return det_box_model, ocr_model, craft, refine_net, args, ocr_model_1
