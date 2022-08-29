epoch=latest.pth
confifle=rects_ae_textspotter_r50_1x.py
path=../work_dirs/rects_ae_textspotter_r50_1x

../tools/rects_dist_test.sh \
    $confifle \
    $path/$epoch \
    1 \
    --json_out $path/results.json
