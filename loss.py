from math import sqrt


class ImageOutput:
    def __init__(self, y, S, B, C):
        self.cells = list()
        for i in range(S):
            for j in range(S):
                t = y[i][j]
                label_for_this_cell = dict()
                #  t is a vector with length 30
                label_for_this_cell['C'] = t[:C]
                label_for_this_cell['boxes'] = list()
                label_for_this_cell['boxes'].append(dict())
                label_for_this_cell['boxes'].append(dict())
                label_for_this_cell['boxes'][0]['p_obj'] = t[20]
                label_for_this_cell['boxes'][1]['p_obj'] = t[21]
                label_for_this_cell['boxes'][0]['w'] = t[22]
                label_for_this_cell['boxes'][0]['h'] = t[23]
                label_for_this_cell['boxes'][0]['x'] = t[24]
                label_for_this_cell['boxes'][0]['y'] = t[25]
                label_for_this_cell['boxes'][1]['w'] = t[26]
                label_for_this_cell['boxes'][1]['h'] = t[27]
                label_for_this_cell['boxes'][1]['x'] = t[28]
                label_for_this_cell['boxes'][1]['y'] = t[29]
                self.cells.append(label_for_this_cell)


def criterion(y, d):
    # y is the network output for a mini-batch, and dd is the corresponding labels.
    batch_size = len(y)

    sum_loss = 0

    for i in range(batch_size):
        yy = y[i]
        dd = d[i]
        # yy: 7 x 7 x 30
        # dd: 7 x 7 x 25, since dd has only one sub-box for one cell
        sum_loss += get_loss_for_one_image(self, yy, dd)
    return sum_loss / (1.0 * batch_size)


def get_loss_for_one_image(y, d):
    # y is the network output for one image, and d is the corresponding label.
    loss = 0
    S = 7  # S x S cells in one image
    B = 2  # B sub-boxex in one cell
    C = 20  # C classes to classify
    yy = ImageOutput(y, S, B, C)
    dd = ImageOutput(d, S, B, C)

    for i in range(S*S):
        loss += get_loss_for_one_cell(yy.cells[i], dd.cells[i])

    return loss


def get_loss_for_one_cell(y_cell, d_cell):
    # Here is a detailed explanation for loss:
    # https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088

    lambda_coord = 5
    lambda_noobj = 0.5
    classification_loss = 0
    localization_loss = 0
    confidence_loss = 0

    has_object_in_this_cell = False
    if d_cell['boxes'][0]['p_obj'] == 1:
        has_object_in_this_cell = True

    responsible_box = get_responsible_box(y_cell['boxes'], d_cell['boxes'])
    print(responsible_box)

    # localization loss
    if has_object_in_this_cell:
        x_diff = responsible_box['x'] - d_cell['boxes'][0]['x']
        y_diff = responsible_box['y'] - d_cell['boxes'][0]['y']
        localization_loss += lambda_coord * (x_diff ** 2 + y_diff ** 2)

        sqrt_w_diff = sqrt(responsible_box['w']) - sqrt(d_cell['boxes'][0]['w'])
        sqrt_h_diff = sqrt(responsible_box['h']) - sqrt(d_cell['boxes'][0]['h'])
        localization_loss += lambda_coord * (sqrt_w_diff ** 2 + sqrt_h_diff ** 2)

    # confidence loss
    if has_object_in_this_cell:
        tuple_responsible = (responsible_box['x'], responsible_box['y'], responsible_box['w'], responsible_box['h'])
        tuple_groundtruth = (d_cell['boxes'][0]['x'], d_cell['boxes'][0]['y'],
                             d_cell['boxes'][0]['w'], d_cell['boxes'][0]['h'])
        iou = compute_iou(tuple_responsible, tuple_groundtruth)
        C_diff = responsible_box['p_obj'] * iou - d_cell['boxes'][0]['p_obj']
        # confidence = p_obj * iou
        # for ground-truth, the confidence is equal to its p_obj
        confidence_loss += C_diff ** 2
    else:
        # Question here!!! for boxes that don't contain objects, do we need to compute confidence as P_obj * IOU?
        # If so, the confidence for those boxes is always zero, which doesn't make sense for optimization

        # Here, we sum up the confidence loss of all boxes.
        for box in y_cell['boxes']:
            # C_diff = box['p_obj'] - d_cell['boxes'][0]['p_obj'], with d_cell['boxes'][0]['p_obj'] = 0
            C_diff = box['p_obj']
            confidence_loss += lambda_noobj * (C_diff ** 2)

    # classification loss
    if has_object_in_this_cell:
        C = len(y_cell['C'])
        for i in range(C):
            pc_diff = y_cell['C'][i] - d_cell['C'][i]
            classification_loss += (pc_diff ** 2)

    return localization_loss + confidence_loss + classification_loss


def get_responsible_box(y_boxes, d_boxes):
    max_iou = -1
    max_iou_box = None
    d_box = (d_boxes[0]['x'], d_boxes[0]['y'], d_boxes[0]['w'], d_boxes[0]['h'])
    for box in y_boxes:
        this_box = (box['x'], box['y'], box['w'], box['h'])
        iou = compute_iou(this_box, d_box)
        if iou > max_iou:
            max_iou = iou
            max_iou_box = box
    return max_iou_box


def compute_iou(rec1, rec2):
    
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)



'''
    Encode boxes and labels to 7x7x30 tensor. For each area, the 30 len tensor has such structure:
    [ 20(class label) | 1(C) | 1(C) | 4(width, height, center_w, center_h, and all are ratio) | 4(the same) ]
'''

yyy = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
ddd = [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

yy = list()
dd = list()

for i in range(7):
    y = list()
    d = list()
    for j in range(7):
        y.append(yyy)
        d.append(ddd)
    yy.append(y)
    dd.append(d)

# print(len(yy))
# print(len(yy[0]))
# print(len(yy[0][0]))

ret = get_loss_for_one_image(yy, dd)
print(ret)


