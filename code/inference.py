import tvm
import numpy as np
from tvm.contrib import graph_runtime
import cv2
import torch
from utils import decode, decode_landm, PriorBox, nms

cfg = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 320,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

image_path = "F:\\work\\learn\\projects\\demo\\RetinaFace\\x64\\Release\\test.jpeg"
graph_json_path = "./models/retina.json"
libpath = "./models/retina.so"
param_path = "./models/retina.params"
vis_thres = 0.1
confidence_threshold = 0.02
nms_threshold = 0.4
top_k = 5000
keep_top_k = 100

def detect():
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #补边
    size = max(img_raw.shape[0], img_raw.shape[1])
    new_img = np.zeros((size, size, 3))
    new_img[:img_raw.shape[0], :img_raw.shape[1], :] = img_raw
    img = cv2.resize(new_img, (320, 320))
    im_height, im_width, _ = img.shape
    #缩放比例
    scale_x = new_img.shape[1] / im_width
    scale_y = new_img.shape[0] / im_height
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

    img = np.float32(img)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    # 加载模型
    loaded_json = open(graph_json_path).read()
    loaded_lib = tvm.runtime.load_module(libpath) #老的版本使用tvm.module.load
    loaded_params = bytearray(open(param_path, "rb").read())

    # 执行模型推理
    ctx = tvm.cpu()
    device = torch.device("cpu")
    module = graph_runtime.create(loaded_json, loaded_lib, ctx)
    module.load_params(loaded_params)
    module.set_input("input0", img)
    module.run()
    bbox_tensor = module.get_output(0).asnumpy()
    classify_tensor = module.get_output(1).asnumpy()
    landmark_tensor = module.get_output(2).asnumpy()

    #后处理
    bbox_tensor = torch.from_numpy(bbox_tensor)
    landmark_tensor = torch.from_numpy(landmark_tensor)
    classify_tensor = torch.from_numpy(classify_tensor)

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(torch.Tensor(bbox_tensor).squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = classify_tensor.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landmark_tensor.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    landms = landms * scale1
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    for b in dets:
        if b[4] < vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        cv2.rectangle(img_raw, (int(b[0] * scale_x), int(b[1] * scale_y)), (int(b[2] * scale_x), int(b[3] * scale_y)), (0, 0, 255), 2)

        # landms
        cv2.circle(img_raw, (int(b[5]*scale_x), int(b[6]*scale_y)), 1, (0, 0, 255), 4)
        cv2.circle(img_raw, (int(b[7]*scale_x), int(b[8]*scale_y)), 1, (0, 255, 255), 4)
        cv2.circle(img_raw, (int(b[9]*scale_x), int(b[10]*scale_y)), 1, (255, 0, 255), 4)
        cv2.circle(img_raw, (int(b[11]*scale_x), int(b[12]*scale_y)), 1, (0, 255, 0), 4)
        cv2.circle(img_raw, (int(b[13]*scale_x), int(b[14]*scale_y)), 1, (255, 0, 0), 4)
    # save image

    name = "test_res.jpg"
    cv2.imwrite(name, img_raw)


if __name__ == '__main__':
    detect()