import cv2
import numpy as np
import random
import utils

img = np.zeros((300, 400), np.uint8)
dets = np.array([[83, 54, 165, 163, 0.8], [67, 48, 118, 132, 0.5], [91, 38, 192, 171, 0.6]], np.float)


# img_cp = img.copy()
# for box in dets.tolist():  # 显示待测试框及置信度
#     x1, y1, x2, y2, score = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[-1]
#     y_text = int(random.uniform(y1, y2))
#     cv2.rectangle(img_cp, (x1, y1), (x2, y2), (255, 255, 255), 2)
#     cv2.putText(img_cp, str(score), (x2 - 30, y_text), 2, 1, (255, 255, 0))
# cv2.imshow("ori_img", img_cp)

rtn_box = utils.nms(dets[:,:4],dets[:,4], 0.3)  # 0.3为faster-rcnn中配置文件的默认值
cls_dets = dets[rtn_box, :]
print("nms box:", cls_dets)

# img_cp = img.copy()
# for box in cls_dets.tolist():
#     x1, y1, x2, y2, score = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[-1]
#     y_text = int(random.uniform(y1, y2))
#     cv2.rectangle(img_cp, (x1, y1), (x2, y2), (255, 255, 255), 2)
#     cv2.putText(img_cp, str(score), (x2 - 30, y_text), 2, 1, (255, 255, 0))

# cv2.imshow()