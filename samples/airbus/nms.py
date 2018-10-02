import numpy as np
from skimage.morphology import label

def multi_rle_encode(img, **kwargs):
    '''
    Encode connected regions as separated masks
    '''
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2), **kwargs) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k, **kwargs) for k in np.unique(labels[labels>0])]

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def rle_encode(img, min_max_threshold=1e-3, max_mean_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_max_threshold:
        return '' ## no need to encode if it's all zeros
    #print(max_mean_threshold, np.mean(img) , max_mean_threshold)
    if max_mean_threshold and np.mean(img) > max_mean_threshold:
        return '' ## ignore overfilled mask
    
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def resultFilter(r, scores_threshold=0.9, overlapThresh=0.5):
    temp = r['scores']
    filt_ids = temp > scores_threshold
    pick = []
    boxes = r['rois']
    y1 = boxes[:,0]
    x1 = boxes[:,1]
    y2 = boxes[:,2]
    x2 = boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(r['scores'])
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        delete_candidates = np.concatenate(([last],
            np.where(overlap > overlapThresh)[0]))
        idxs = np.delete(idxs, delete_candidates)

    # return only the bounding boxes that were picked using the
    # integer data type
    filt_mask = np.zeros(len(filt_ids), dtype=bool)
    filt_mask[pick] = True
    filt_ids = np.logical_and(filt_ids, filt_mask)
    mask = r['masks'][:,:,filt_ids]
    return mask

def resultFilterMaxScore(r, scores_threshold=0.9, overlapThresh=0.5):
    temp = r['scores']
    filt_ids = temp > scores_threshold
    pick = []
    boxes = r['rois']
    y1 = boxes[:,0]
    x1 = boxes[:,1]
    y2 = boxes[:,2]
    x2 = boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        delete_candidates = np.concatenate(([last],
            np.where(overlap > overlapThresh)[0]))
        idxs = np.delete(idxs, delete_candidates)
        
        curr_score = 0
        nms_index = -1
        for candidate in delete_candidates:
            if temp[candidate] > curr_score:
                nms_index = candidate
        print(nms_candidate)
        pick.append(nms_candidate)

    # return only the bounding boxes that were picked using the
    # integer data type
    print(pick)
    #filt_mask = np.zeros(len(filt_ids), dtype=bool)
    #filt_mask[pick] = True
    #filt_ids = np.logical_and(filt_ids, filt_mask)
    #mask = r['masks'][:,:,filt_ids]
    #return mask

def verfy_encoding(encoding):
    if type(encoding) is str:
        #print(type(row['EncodedPixels']) is str)
        s = encoding.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        sorted_starts = sorted(starts)
        not_ascending = sorted_starts != starts
        if not_ascending.any():
            print('ascend')
            #if index == 1:
            #    print(row['EncodedPixels'])
            print(index)
        overlaps = starts + lengths
        possible_overlap = overlaps[:-1] - sorted_starts[1:]
        is_overlapping = possible_overlap >= 0
        if is_overlapping.any():
            print('overlap')
            #print(row['EncodedPixels'])
            print(index)
