from math import sqrt


def criterion(self, y, d):
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


def get_loss_for_one_image(self, y, d):
    # y is the network output for one image, and d is the corresponding label.
    loss = 0
    S = 7  # S x S cells in one image
    B = 2  # B sub-boxex in one cell
    C = 20  # C classes to classify
    yy = ImageOutput(y, S, B, C)
    dd = ImageOutput(d, S, 1, C)

    for i in range(S*S):
        loss += get_loss_for_one_cell(yy.cells[i], dd.cells[i])

    return loss


def get_loss_for_one_cell(self, y_cell, d_cell):
    # Here is a detailed explanation for loss:
    # https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088

    lambda_coord = 5
    lambda_noobj = 0.5
    classification_loss = 0
    localization_loss = 0
    confidence_loss = 0

    has_object_in_this_cell = False
    if d_cell['boxes'][0]['confidence'] == 1:
        has_object_in_this_cell = True

    responsible_box = get_responsible_box(y_cell['boxes'], d_cell['boxes'])

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
        C_diff = responsible_box['confidence'] - d_cell['boxes'][0]['confidence']
        confidence_loss += C_diff ** 2
    else:
        C_diff = responsible_box['confidence'] - d_cell['boxes'][0]['confidence']
        confidence_loss += lambda_noobj * (C_diff ** 2)

    # classification loss
    if has_object_in_this_cell:
        C = len(y_cell['C'])
        for i in range(C):
            pc_diff = y_cell['C'][i] - d_cell['C'][i]
            classification_loss += (pc_diff ** 2)

    return localization_loss + confidence_loss + classification_loss


def get_responsible_box(self, y_boxes, d_boxes):
    max_iou = 0
    max_iou_box = None
    d_box = (d_boxes[0]['x'], d_boxes[0]['y'], d_boxes[0]['w'], d_boxes[0]['h'])
    for box in y_boxes:
        this_box = (box['x'], box['y'], box['w'], box['h'])
        IOU = compute_iou(this_box, d_box)
        if IOU > max_iou:
            max_iou = IOU
            max_iou_box = box
    return max_iou_box


def compute_iou(box1, box2):
    pass


class ImageOutput:
    def __init__(self, y, S, B, C):
        self.cells = list()
        for i in range(S):
            for j in range(S):
                t = y[i][j]
                label_for_this_cell = dict()
                #  t is a vector with length 30
                for k in range(B):
                    d = dict()
                    d['confidence'] = t[k*5]
                    d['x'] = t[k*5 + 1]
                    d['y'] = t[k*5 + 2]
                    d['w'] = t[k * 5 + 3]
                    d['h'] = t[k * 5 + 4]
                    label_for_this_cell['boxes'].append(d)
                label_for_this_cell['C'] = y[5*B:]
                self.cells.append(label_for_this_cell)
