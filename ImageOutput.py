class ImageOutput:
    def __init__(self, y, S, B, C):
        self.cells = list()
        for i in range(S):
            for j in range(S):
                t = y[i][j]
                label_for_this_cell = dict()
                #  t is a vector with length 30
                label_for_this_cell['C'] = y[:C]
                label_for_this_cell['boxes'][0] = dict()
                label_for_this_cell['boxes'][1] = dict()
                label_for_this_cell['boxes'][0]['p_obj'] = y[20]
                label_for_this_cell['boxes'][1]['p_obj'] = y[21]
                label_for_this_cell['boxes'][0]['w'] = y[22]
                label_for_this_cell['boxes'][0]['h'] = y[23]
                label_for_this_cell['boxes'][0]['x'] = y[24]
                label_for_this_cell['boxes'][0]['y'] = y[25]
                label_for_this_cell['boxes'][1]['w'] = y[26]
                label_for_this_cell['boxes'][1]['h'] = y[27]
                label_for_this_cell['boxes'][1]['x'] = y[28]
                label_for_this_cell['boxes'][1]['y'] = y[29]
                self.cells.append(label_for_this_cell)