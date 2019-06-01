def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :param rec: (width, height, centere_w, center_h)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    width1 = rec1[0]
    height1 = rec1[1]
    center_w1 = rec1[2]
    center_h1 = rec1[3]
    width2 = rec2[0]
    height2 = rec2[1]
    center_w2 = rec2[2]
    center_h2 = rec2[3]

    rec1 = (center_h1-0.5*height1, center_w1-0.5*width1, center_h1+0.5*height1, center_w1+0.5*width1)
    rec2 = (center_h2-0.5*height2, center_w2-0.5*width2, center_h2+0.5*height2, center_w2+0.5*width2)

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