#CUDA_VISIBLE_DEVICES=5 python tools/test.py configs/cartons/faster_rcnn_r50_caffe_fpn_1x_cartons.py work_dirs/faster_rcnn_r50_caffe_fpn_1x_cartons/latest.pth --eval bbox segm --show-dir results/faster_rcnn_r50_caffe_fpn_1x_cartons
CUDA_VISIBLE_DEVICES=5 python tools/test.py configs/cartons/mask_rcnn_r50_fpn_poly_1x_coco.py work_dirs/mask_rcnn_r50_fpn_poly_1x_coco/latest.pth --eval bbox segm --show-dir results/mask_rcnn_r50_fpn_poly_1x_coco