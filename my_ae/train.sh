#多卡训练
#step 1
../tools/rects_dist_train.sh \
    rects_ae_textspotter_r50_1x.py \
    4 \
    --validate \
    --autoscale-lr \
    # --work_dir ../work_dirs/rects_ae_textspotter_r50_1x/
    # --resume_from

#step 2
#训 language model
# tools/rects_dist_train.sh \
#     local_configs/rects_ae_textspotter_lm_r50_1x.py \
#     8

