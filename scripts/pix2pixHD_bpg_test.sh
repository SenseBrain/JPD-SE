# get test results
for q in {51,48,45,42}; 
  do python test.py --model pix2pixHD --dataset cityscapes --load_opt --opt_file /path/to/checkpoints/pix_bpgq${q}_1024/opt.pkl --checkpoints_dir /path/to/checkpoints/pix_bpgq${q}_1024/ --do_not_get_codes --gpu_ids 6 --save_dir /path/to/output/q${q} --root_dir datasets/cityscapes_test_CVPR20_1024;
done;