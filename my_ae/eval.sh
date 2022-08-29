epoch=latest.pth
confile=rects_ae_textspotter_r50_1x.py
path=../work_dirs/rects_ae_textspotter_r50_1x
logfile=eval.txt
../tools/rects_dist_test.sh \
    $confile \
    $path/$epoch \
    4 \
    --json_out $path/results.json
echo $epoch >> $logfile
python -u ../tools/rects_eval.py \
    --cfg_file $confile \
    --pred_file $path/results.json \
    | tee -a $logfile