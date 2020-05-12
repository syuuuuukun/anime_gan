#　Stylegan2生成結果

[参考にしたコード](https://github.com/rosinality/stylegan2-pytorch)

![80000iter](https://github.com/syuuuuukun/anime_gan/blob/master/gans/stylegan2/imgs/80000.png)

## 問題点
手とか背景と物体の境界にボワボワがある

## 試みたこと
試したこと
（セグメンテーション画像みたいなのを入れたら解決できるのではないか）

Generator: seg画像 -> 5層Conv2d -> pose_x -> (pose_x + noise)　-> Ganerator -> Fake画像
Discriminator: 画像 -> Discriminator -> 画像embedding + seg画像embedding -> adv_loss
(projectionGANみたいな)

生成はできるけど，手やぼやぼやは消えない

## 今後やるとしたら
lossで改善する場合: 画像と背景の間の境界をsegmentation系の手法でうまく対処できるかも
http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Learning_Active_Contour_Models_for_Medical_Image_Segmentation_CVPR_2019_paper.pdf

ネットワークで改善する場合: そもそももっとStyleGanのlayer数と潜在変数を増やしたら細かいところも生成できるかも