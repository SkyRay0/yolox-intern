import os
import av
import sys
import argparse
import time
from loguru import logger
import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False,
            legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def parse_image(_predictor, result_path, current_time, path):
    outputs, img_info = _predictor.inference(path)
    result_image = _predictor.visual(outputs[0], img_info, _predictor.confthre)
    #save_folder = os.path.join(
    #    vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    #)
    #os.makedirs(save_folder, exist_ok=True)
    #save_file_name = os.path.join(save_folder, os.path.basename(path))
    #logger.info("Saving detection result in {}".format(save_file_name))
    cv2.imwrite(result_path, result_image)
    ch = cv2.waitKey(0)
    if ch == 27 or ch == ord("q") or ch == ord("Q"):
        print("kys")


def parse_video(_predictor, result_path, current_time, path):
    cap = cv2.VideoCapture(path)
    logger.info(f"Successfully opened video file at {path}")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    # save_folder = os.path.join(
    # vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    # )
    # os.makedirs(save_folder, exist_ok=True)
    # save_path = os.path.join(save_folder, os.path.basename(path))
    # logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        result_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps,
        (int(width), int(height))
    )
    logger.info(f"Ready to write new video file at {result_path}")
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = _predictor.inference(frame)
            result_frame = _predictor.visual(outputs[0], img_info, _predictor.confthre)
            vid_writer.write(result_frame)
            #cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
            #cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


parser = argparse.ArgumentParser()
parser.add_argument('--frames', default="100,200,224")
parser.add_argument('-i')
parser.add_argument('--model')
parser.add_argument('--model_path')
namespace = parser.parse_args()
frames = namespace.frames.split(",")

exp = get_exp(None, namespace.model)
model = exp.get_model()
model.eval()
ckpt = torch.load(namespace.model_path, map_location="cpu")
model.load_state_dict(ckpt["model"])
predictor = Predictor(model, exp, COCO_CLASSES, None, None, "cpu", False, False)
logger.info("Successfully loaded model yolox-x")

container = av.open(namespace.i)
stream = container.streams.video[0]
container_decoded = list(container.decode(stream))
logger.info(f"Frames total: {len(container_decoded)}")

for i in 0, 1, 2:
    container_decoded[int(frames[i])].to_image().save(
        f"\\intern_script\\frame-{frames[i]}.jpg")
    parse_image(predictor,
                f"\\intern_script\\frame-{frames[i]}-parsed.jpg",
                time.localtime(), f"\\intern_script\\frame-{frames[i]}.jpg")

parse_video(predictor, "\\intern_script\\develop-streem-parsed.mp4",
            time.localtime(), namespace.i)
