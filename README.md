# SoccerNetBall_2

### Notes
1. data/soccernetball/labels_path.txt 需要修改
2. config 中
- SoccerNetBall_rny002gsf_bs4_clip120_FreezeNewEncoder: 使用的是 Unisoccer pretrain 出来的 siglip image encoder 并 freeze - jyrao.github.io/UniSoccer/ 。会使用对应的train - train_tdeed_bas_newencoder.py 
- SoccerNetBall_rny002gsf_bs4_FreezePretrainedEncoder: 使用的是官方提供的 baseline encoder 并 freeze。会使用对应的train - train_tdeed_bas_pretrainedencoder.py
- SoccerNetBall_rny002gsf_bs4_twomlps: 将 队伍和分类的分类头合并到一起，使用对应的train - train_tdeed_bas_twomlps.py
- 其他就是简单的测试： 1. 120长度的clip而不是100，原因是 Unisoccer pretrained 的模型是以30帧为单位的，所以使用了 clip120。2. 008 的 backbone。3. random seed=42。

### Todo
1. Verify the validation phase to check if it is random or not. If it is random, try to fix it.
2. Merge ordinal regression and check the performance.
3. Check regmixup and implement it to the pipeline.
4. Figure out the reason why the val losses show different trend compared to mAP on val.