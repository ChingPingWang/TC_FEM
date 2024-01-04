import numpy as np

def box_filp_trans(box_list,mode,h,w):
    if mode=='fliplr':
      flip_matrix = np.array([[-1,0],[0,1]])
    if mode=='flipud':
      flip_matrix = np.array([[1,0],[0,-1]])
    if mode=='fliplrud':
      flip_matrix = np.array([[-1,0],[0,-1]])
    new_box_list = []
    for ii in range(len(box_list)):
      bx1, by1, bx2, by2 = box_list[ii][0], box_list[ii][1], box_list[ii][2], box_list[ii][3]
      p1, p2 = np.array([bx1,by1]), np.array([bx2,by2])
      if mode=='fliplr':
        nbx1, nby1 = flip_matrix.dot(p2)[0]+w, flip_matrix.dot(p1)[1]
        nbx2, nby2 = flip_matrix.dot(p1)[0]+w, flip_matrix.dot(p2)[1]
      if mode=='flipud':
        nbx1, nby1 = flip_matrix.dot(p1)[0], flip_matrix.dot(p2)[1]+h
        nbx2, nby2 = flip_matrix.dot(p2)[0], flip_matrix.dot(p1)[1]+h
      if mode=='fliplrud':
        nbx1, nby1 = flip_matrix.dot(p2)[0]+w, flip_matrix.dot(p2)[1]+h
        nbx2, nby2 = flip_matrix.dot(p1)[0]+w, flip_matrix.dot(p1)[1]+h
      new_box_list.append([nbx1, nby1, nbx2, nby2])
    return new_box_list


box_list = np.array([[5,10,5,10],[5,10,5,10]])
print(box_list)
result = box_filp_trans(box_list,'fliplr',20,10)
print(result)