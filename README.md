# ganのりぽじとり

## dcgan_6464
64x64の画像生成
### Generator
deconv -> BN -> LeakyReLU　を4層
### Discriminator
conv -> BN -> LeakyReLU を4層
### loss
hinge-loss\
0-gploss

## resgan
256x256の画像生成
### Generator
6層のresblock -> brc -> Tanh\
nn.UpsamplingNearest2d -> convで画像の拡大

### Discriminator
6層のresblock -> Conv2d\
nn.UpsamplingNearest2d -> convで画像の縮小\
ResblockにはBNを使わない，活性化関数にはLeakyReLU

### loss
hinge-loss\
0-gploss

### その他
weightの初期化はnn.init.xavier_normal_(m.weight)\
DiscriminatorとGeneratorの更新比率は2:1\
Adam lr=0.0002,β1=0.5,β2=0.99


