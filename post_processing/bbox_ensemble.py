# from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import numpy as np
# from numba import jit
import os


def prepare_boxes(boxes, scores, labels, masks):
    result_boxes = boxes.copy()

    cond = (result_boxes < 0)
    cond_sum = cond.astype(np.int32).sum()
    if cond_sum > 0:
        print('Warning. Fixed {} boxes coordinates < 0'.format(cond_sum))
        result_boxes[cond] = 0

    cond = (result_boxes > 1)
    cond_sum = cond.astype(np.int32).sum()
    if cond_sum > 0:
        print('Warning. Fixed {} boxes coordinates > 1. Check that your boxes was normalized at [0, 1]'.format(cond_sum))
        result_boxes[cond] = 1

    boxes1 = result_boxes.copy()
    result_boxes[:, 0] = np.min(boxes1[:, [0, 2]], axis=1)
    result_boxes[:, 2] = np.max(boxes1[:, [0, 2]], axis=1)
    result_boxes[:, 1] = np.min(boxes1[:, [1, 3]], axis=1)
    result_boxes[:, 3] = np.max(boxes1[:, [1, 3]], axis=1)

    area = (result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1])
    cond = (area == 0)
    cond_sum = cond.astype(np.int32).sum()
    if cond_sum > 0:
        print('Warning. Removed {} boxes with zero area!'.format(cond_sum))
        result_boxes = result_boxes[area > 0]
        scores = scores[area > 0]
        labels = labels[area > 0]
        masks = masks[area > 0]

    return result_boxes, scores, labels, masks


def cpu_soft_nms_float(dets, sc, Nt, sigma, thresh, method):
    """
    Based on: https://github.com/DocF/Soft-NMS/blob/master/soft_nms.py
    It's different from original soft-NMS because we have float coordinates on range [0; 1]
    :param dets:   boxes format [x1, y1, x2, y2]
    :param sc:     scores for boxes
    :param Nt:     required iou 
    :param sigma:  
    :param thresh: 
    :param method: 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS
    :return: index of boxes to keep
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1, x1, y2, x2]
    y1 = dets[:, 1]
    x1 = dets[:, 0]
    y2 = dets[:, 3]
    x2 = dets[:, 2]
    scores = sc
    areas = (x2 - x1) * (y2 - y1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N - 1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)
    return keep


# @jit(nopython=True)
def nms_float_fast(dets, scores, thresh):
    """
    # It's different from original nms because we have float coordinates on range [0; 1]
    :param dets: numpy array of boxes with shape: (N, 5). Order: x1, y1, x2, y2, score. All variables in range [0; 1]
    :param thresh: IoU value for boxes
    :return: index of boxes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nms_method(boxes, scores, labels, masks, method=3, iou_thr=0.5, sigma=0.5, thresh=0.001, weights=None):
    """
    :param boxes: list of boxes predictions from each model, each box is 4 numbers. 
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1] 
    :param scores: list of scores for each model 
    :param labels: list of labels for each model
    :param method: 1 - linear soft-NMS, 2 - gaussian soft-NMS, 3 - standard NMS
    :param iou_thr: IoU value for boxes to be a match 
    :param sigma: Sigma value for SoftNMS
    :param thresh: threshold for boxes to keep (important for SoftNMS)
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2). 
    :return: scores: confidence scores
    :return: labels: boxes labels
    """

    # If weights are specified
    if weights is not None:
        if len(boxes) != len(weights):
            print('Incorrect number of weights: {}. Must be: {}. Skip it'.format(len(weights), len(boxes)))
        else:
            weights = np.array(weights)
            for i in range(len(weights)):
                scores[i] = (np.array(scores[i]) * weights[i]) / weights.sum()

    # We concatenate everything
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    masks = np.concatenate(masks)

    # Fix coordinates and removed zero area boxes
    boxes, scores, labels, masks = prepare_boxes(boxes, scores, labels, masks)
    # print(labels.shape)
    fk_lables = np.zeros(labels.shape)

    # Run NMS independently for each label
    # unique_labels = np.unique(labels)
    # unique_labels = np.unique(fk_lables)
    final_boxes = []
    final_scores = []
    final_labels = []
    final_masks = []
    # for l in unique_labels:
    #     # condition = (labels == l)
    #     condition = (fk_lables == l)
    #     boxes_by_label = boxes[condition]
    #     scores_by_label = scores[condition]
    #     masks_by_label = masks[condition]
    #     # labels_by_label = np.array([l] * len(boxes_by_label))
    #     labels_by_label = labels[condition]

    #     if method != 3:
    #         keep = cpu_soft_nms_float(boxes_by_label.copy(), scores_by_label.copy(), Nt=iou_thr, sigma=sigma, thresh=thresh, method=method)
    #     else:
    #         # Use faster function
    #         keep = nms_float_fast(boxes_by_label, scores_by_label, thresh=iou_thr)

    #     final_boxes.append(boxes_by_label[keep])
    #     final_scores.append(scores_by_label[keep])
    #     final_labels.append(labels_by_label[keep])
    #     final_masks.append(masks_by_label[keep])

    condition = (fk_lables == 0)
    boxes_by_label = boxes[condition]
    scores_by_label = scores[condition]
    masks_by_label = masks[condition]
    labels_by_label = labels[condition]

    if method != 3:
        keep = cpu_soft_nms_float(boxes_by_label.copy(), scores_by_label.copy(), Nt=iou_thr, sigma=sigma, thresh=thresh, method=method)
    else:
        # Use faster function
        keep = nms_float_fast(boxes_by_label, scores_by_label, thresh=iou_thr)

    final_boxes.append(boxes_by_label[keep])
    final_scores.append(scores_by_label[keep])
    final_labels.append(labels_by_label[keep])
    final_masks.append(masks_by_label[keep])

    final_boxes = np.concatenate(final_boxes)
    final_scores = np.concatenate(final_scores)
    final_labels = np.concatenate(final_labels)
    final_masks = np.concatenate(final_masks)

    return final_boxes, final_scores, final_labels, final_masks


def nms(boxes, scores, labels, masks, iou_thr=0.5, weights=None):
    """
    Short call for standard NMS 
    
    :param boxes: 
    :param scores: 
    :param labels: 
    :param iou_thr: 
    :param weights: 
    :return: 
    """
    return nms_method(boxes, scores, labels, masks, method=3, iou_thr=iou_thr, weights=weights)


def soft_nms(boxes, scores, labels, method=2, iou_thr=0.5, sigma=0.5, thresh=0.001, weights=None):
    """
    Short call for Soft-NMS
     
    :param boxes: 
    :param scores: 
    :param labels: 
    :param method: 
    :param iou_thr: 
    :param sigma: 
    :param thresh: 
    :param weights: 
    :return: 
    """
    return nms_method(boxes, scores, labels, method=method, iou_thr=iou_thr, sigma=sigma, thresh=thresh, weights=weights)

def mmEnsemble(scores_list, bboxes_list, masks_list, lable_list, thr):

    # scores_list = list()
    # bboxes_list = list()
    # masks_list = list()
    # lable_list = list()

    # for i in range(6):
    #     temp_sc = np.concatenate([result_1152[0][i][:,-1], result_1024[0][i][:,-1], result_800[0][i][:,-1], result_512[0][i][:,-1], result_256[0][i][:,-1]])
    #     temp_bb = np.concatenate([result_1152[0][i][:,0:4], result_1024[0][i][:,0:4], result_800[0][i][:,0:4], result_512[0][i][:,0:4], result_256[0][i][:,0:4]])
    #     temp_mk = result_1152[1][i] + result_1024[1][i] + result_800[1][i] + result_512[1][i] + result_256[1][i]

    #     for j in range(len(temp_mk)):
    #         scores_list.append(temp_sc[j])
    #         bboxes_list.append(temp_bb[j, 0:4])
    #         masks_list.append(temp_mk[j])
    #         lable_list.append(i)

    scores = np.array(scores_list)
    bboxes = np.array(bboxes_list)[scores>thr]
    masks = np.array(masks_list)[scores>thr]
    lables = np.array(lable_list)[scores>thr]
    scores = scores[scores>thr]

    # print(scores.shape)
    # print(bboxes.shape)
    # print(masks.shape)
    # print(lables.shape)
    # if len(masks) == 0:
    #     return result_1152
    # else:

    iou_thr = 0.1
    skip_box_thr = 0.0001
    sigma = 0.1
    
    boxes, scores, labels, masks = nms([bboxes/1024], [scores], [lables], [masks], weights=None, iou_thr=iou_thr)

    boxes = boxes*1024
    print('yes')
    # pdb.set_trace()
    # boxes = np.column_stack([boxes, scores])

    # for i in boxes:
    #     print(i)
    # cl_bb_list = list()
    # cl_mk_list = list()

    # for clss in range(6):
    #     con = labels == clss

    #     if con.sum() == 0:
    #         bb = np.zeros([0,5])
    #         mk = []
    #     else:
    #         bb = boxes[con]
    #         mk = masks[con]
    #     cl_bb_list.append(bb)
    #     cl_mk_list.append(mk)
    # ensemble_result = (cl_bb_list, cl_mk_list)
    return boxes, scores, labels, masks
    # return bboxes, scores, lables, masks

def trans_format(result):
    box_ls = result[0].tolist()
    sco_ls = result[1]
    cla_ls = result[2]
    msk_ls = result[3]
    trans_result,nb_ls = [],[]
    for i in range(len(box_ls)):
        x1, y1, x2, y2 = int(box_ls[i][0]), int(box_ls[i][1]), int(box_ls[i][2]), int(box_ls[i][3])
        nb_ls = [x1,x2,y1,y2]
        trans_result.append([cla_ls[i],nb_ls,msk_ls[i],sco_ls[i]])
        # pdb.set_trace()
    return trans_result

def trans_bb_format(box_ls):
    box_ls = box_ls.tolist()
    nb_ls = []
    for i in range(len(box_ls)):
        x1, x2, y1, y2 = int(box_ls[i][0]), int(box_ls[i][1]), int(box_ls[i][2]), int(box_ls[i][3])
        nb_ls.append([x1,y1,x2,y2])#nms format
        # pdb.set_trace()
    return nb_ls

if __name__ == '__main__':

    # device = 'cuda:0'
    # # checkpoint_file = 'epoch_45_aug.pth'
    # checkpoint_file = 'epoch_41_1024.pth'

    # images_path = 'E:/coince/code/swim_test/multi_resolution_test/npy_data/shineV3_test_img.npy'
    # labels_path = 'E:/coince/code/swim_test/multi_resolution_test/npy_data/shineV3_test_lab.npy'
    # img_chunk = np.load(images_path).astype(np.uint8)
    # print(img_chunk.shape)

    # class_tag = ['neutrophil', 'epithelial-cell', 'lymphocyte', 'plasma-cell', 'eosinophil', 'connective-tissue-cell']

    # pred_regression = {
    #     class_tag[0] : list(),
    #     class_tag[1] : list(),
    #     class_tag[2] : list(),
    #     class_tag[3] : list(),
    #     class_tag[4] : list(),
    #     class_tag[5] : list()
    # }

    # test_image = img_chunk[3, :, :, :]
    
    # # model = init_detector(config_file, checkpoint_file, device=device)
    # # model_1280 = init_detector('swin_HTC_config_1280.py', checkpoint_file, device=device)
    # model_1152 = init_detector('swin_HTC_config_1152.py', checkpoint_file, device=device)
    # model_1024 = init_detector('swin_HTC_config_1024.py', checkpoint_file, device=device)
    # model_800 = init_detector('swin_HTC_config_800.py', checkpoint_file, device=device)
    # model_512 = init_detector('swin_HTC_config_512.py', checkpoint_file, device=device)
    # model_256 = init_detector('swin_HTC_config_256.py', checkpoint_file, device=device)

    import pdb
    # result_1152 = inference_detector(model_1152, test_image) 
    # result_1024 = inference_detector(model_1024, test_image)
    # result_800 = inference_detector(model_800, test_image)
    # result_512 = inference_detector(model_512, test_image)
    # result_256 = inference_detector(model_256, test_image)

    # im_path = 'G:/PAIP_2023/data/PAIP2023_Test/' #final test
    im_path = 'G:/PAIP_2023/data/PAIP_in_test/' #inside test
    im_pls = os.listdir(im_path)

    for ii in im_pls:
        id_name = ii[:-4]
        # print(id_name)
        test_bb = np.load('G:/PAIP_2023/tta/mrcnn/test_111_luo/'+id_name+'_bb.npy',allow_pickle=True)
        test_cl = np.load('G:/PAIP_2023/tta/mrcnn/test_111_luo/'+id_name+'_cl.npy',allow_pickle=True)
        test_mk = np.load('G:/PAIP_2023/tta/mrcnn/test_111_luo/'+id_name+'_mk.npy',allow_pickle=True)
        test_sc = np.load('G:/PAIP_2023/tta/mrcnn/test_111_luo/'+id_name+'_sc.npy',allow_pickle=True)
        
        # test_bb_ls = test_bb.tolist()
        # test_cl_ls = test_cl.tolist()
        # test_mk_ls = test_mk.tolist()
        # test_sc_ls = test_sc.tolist()
        
        test_bb_new = trans_bb_format(test_bb)
        result = mmEnsemble(test_sc, test_bb_new, test_mk, test_cl, thr=0.7)
        new_r = trans_format(result)
        
        save_path = 'G:/PAIP_2023/tta/result/mrcnn/val_111_tta_luo_npy/'+id_name+'.npy'
        np.save(save_path,new_r)
        

    # show_result_pyplot(model_1024, test_image, result)
    # show_result_pyplot(model_1024, test_image, result_1152)
    # show_result_pyplot(model_1024, test_image, result_1024)
    # show_result_pyplot(model_1024, test_image, result_800)
    # show_result_pyplot(model_1024, test_image, result_512)
    # show_result_pyplot(model_1024, test_image, result_256)

    # scores_list = list()
    # bboxes_list = list()
    # masks_list = list()
    # lable_list = list()

    # for i in range(6):
    #     temp_sc = np.concatenate([result_1152[0][i][:,-1], result_1024[0][i][:,-1], result_800[0][i][:,-1], result_512[0][i][:,-1], result_256[0][i][:,-1]])
    #     temp_bb = np.concatenate([result_1152[0][i][:,0:4], result_1024[0][i][:,0:4], result_800[0][i][:,0:4], result_512[0][i][:,0:4], result_256[0][i][:,0:4]])
    #     temp_mk = result_1152[1][i] + result_1024[1][i] + result_800[1][i] + result_512[1][i] + result_256[1][i]

    #     for j in range(len(temp_mk)):
    #         scores_list.append(temp_sc[j])
    #         bboxes_list.append(temp_bb[j, 0:4])
    #         masks_list.append(temp_mk[j])
    #         lable_list.append(i)

    # scores = np.array(scores_list)
    # bboxes = np.array(bboxes_list)
    # masks = np.array(masks_list)
    # lables = np.array(lable_list)

    # print(scores.shape)
    # print(bboxes.shape)
    # print(masks.shape)
    # print(lables.shape)

    # iou_thr = 0.5
    # skip_box_thr = 0.0001
    # sigma = 0.1

    # boxes, scores, labels, masks = nms([bboxes/256], [scores], [lables], [masks], weights=None, iou_thr=iou_thr)
    # boxes = boxes *256
    # boxes = np.column_stack([boxes, scores])

    # # for i in boxes:
    # #     print(i)

    # cl_bb_list = list()
    # cl_mk_list = list()

    # for clss in range(6):
    #     con = labels == clss

    #     if con.sum() == 0:
    #         bb = np.zeros([0,5])
    #         mk = []
    #     else:
    #         bb = boxes[con]
    #         mk = masks[con]
    #     cl_bb_list.append(bb)
    #     cl_mk_list.append(mk)
    # ensemble_result = (cl_bb_list, cl_mk_list)
    # show_result_pyplot(model_1024, test_image, result_1152)
