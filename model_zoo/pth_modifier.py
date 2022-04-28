import sys
import copy
import torch
import fire


def mod(pth_dir):
    pth_dict = torch.load(pth_dir)
    key_list = copy.deepcopy(list(pth_dict['state_dict'].keys()))
    for key in key_list:
        pth_dict['state_dict']['img_' + key] = pth_dict['state_dict'].pop(key)
    torch.save(pth_dict, '%s_mod.pth' % pth_dir[:-4])
    print('%s converting complete' % pth_dir)


def union(pts_dir, img_dir):
    pts_dict = torch.load(pts_dir)
    img_dict = torch.load(img_dir)
    pts_key = copy.deepcopy(list(pts_dict['state_dict'].keys()))
    img_key = copy.deepcopy(list(img_dict['state_dict'].keys()))
    for key in img_key:
        pts_dict['state_dict']['img_' + key] = img_dict['state_dict'].pop(key)
    torch.save(pts_dict, '%s_union.pth' % pts_dir[:-4])
    print('%s converting complete' % pts_dir)

if __name__ == '__main__':
    fire.Fire()
