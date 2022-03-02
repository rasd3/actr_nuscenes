#CUDA_VISIBLE_DEVICES=2 python3 tools/train.py configs/mvxnet/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_nus-3d-3class.py

#CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/mvxnet/dv_mvx-fpn_center_img_secfpn_adamw_2x8_80e_nus-3d-3class.py 2
#CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/nuimages/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim.py 2
CUDA_VISIBLE_DEVICES=0,1 tools/dist_test.sh configs/nuimages/htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim.py htc_x101_64x4d_fpn_dconv_c3-c5_coco-20e_16x1_20e_nuim_20201008_211222-0b16ac4b.pth 2 --show --show-dir htc_nuimage
#CUDA_VISIBLE_DEVICES=0,1 tools/dist_test.sh configs/mvxnet/dv_mvx-fpn_center_secfpn_adamw_2x8_80e_nus-3d-3class.py work_dirs/dv_mvx-fpn_center_secfpn_adamw_2x8_80e_nus-3d-3class/epoch_40.pth 2 --eval bbox

python3 alarm.py
