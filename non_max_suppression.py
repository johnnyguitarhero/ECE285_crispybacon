def non_max_suppression(boxes, scores, threshold):
    """执行non-maximum suppression并返回保留的boxes的索引.
    boxes: [N, (y1, x1, y2, x2)].注意(y2, x2)可以会超过box的边界.
    scores: box的分数的一维数组.
    threshold: Float型. 用于过滤IoU的阈值.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)
 
    #计算box面积
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)
 
    #获取根据分数排序的boxes的索引(最高的排在对前面)
    ixs = scores.argsort()[::-1]
 
    pick = []
    while len(ixs) > 0:
        #选择排在最前的box，并将其索引加到列表中
        i = ixs[0]
        pick.append(i)
        #计算选择的box与剩下的box的IoU
        iou = compute_iou(boxes[i], boxes[ixs[1:]], 
            area[i], area[ixs[1:]])
        #确定IoU大于阈值的boxes. 这里返回的是ix[1:]之后的索引，
        #所以为了与ixs保持一致，将结果加1
        remove_ixs = np.where(iou > threshold)[0] + 1
        #将选择的box和重叠的boxes的索引删除.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)