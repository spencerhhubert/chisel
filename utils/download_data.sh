#!/bin/bash

#run after docker environment has been setup
#downloads and processes abc dataset
#should probably run in nohup because it takes a long time
#each unachieved file is also like 50gb

CONTAINER_ID=$1

mkdir -p data && mkdir -p data/ABC-Dataset

dir="data/ABC-Dataset"
cd $dir
pwd
echo pwd

cat > obj_v00.txt << EOF
https://archive.nyu.edu/rest/bitstreams/89085/retrieve abc_0000_obj_v00.7z
https://archive.nyu.edu/rest/bitstreams/89088/retrieve abc_0001_obj_v00.7z
https://archive.nyu.edu/rest/bitstreams/89091/retrieve abc_0002_obj_v00.7z
https://archive.nyu.edu/rest/bitstreams/89094/retrieve abc_0003_obj_v00.7z
https://archive.nyu.edu/rest/bitstreams/89097/retrieve abc_0004_obj_v00.7z
https://archive.nyu.edu/rest/bitstreams/89100/retrieve abc_0005_obj_v00.7z
https://archive.nyu.edu/rest/bitstreams/89103/retrieve abc_0006_obj_v00.7z
https://archive.nyu.edu/rest/bitstreams/89106/retrieve abc_0007_obj_v00.7z
EOF

echo ls

cat obj_v00.txt | xargs -n 2 -P 8 /bin/bash -c 'wget --no-check-certificate $0 -O $1'
rm obj_v00.txt

sudo apt update && sudo apt install -y p7zip-rar

for file in ./*.7z; do
    filename=$(basename "$file" .7z)
    mkdir "$filename"
    7z x "$file" -o"$filename"
done

#process data
sudo docker run -v /mnt/red/ABC-Dataset/:/chisel/data/ABC-Dataset --gpus all $CONTAINER_ID sh -c "python ABCDataset.py" 

# more files
# https://archive.nyu.edu/rest/bitstreams/89109/retrieve abc_0008_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89112/retrieve abc_0009_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89115/retrieve abc_0010_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89118/retrieve abc_0011_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89121/retrieve abc_0012_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89124/retrieve abc_0013_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89127/retrieve abc_0014_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89130/retrieve abc_0015_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89136/retrieve abc_0016_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89139/retrieve abc_0017_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89142/retrieve abc_0018_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89145/retrieve abc_0019_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89148/retrieve abc_0020_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89151/retrieve abc_0021_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89154/retrieve abc_0022_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89157/retrieve abc_0023_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89160/retrieve abc_0024_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89163/retrieve abc_0025_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89166/retrieve abc_0026_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89169/retrieve abc_0027_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89172/retrieve abc_0028_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89175/retrieve abc_0029_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89178/retrieve abc_0030_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89181/retrieve abc_0031_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89184/retrieve abc_0032_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89187/retrieve abc_0033_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89191/retrieve abc_0034_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89194/retrieve abc_0035_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89197/retrieve abc_0036_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89200/retrieve abc_0037_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89203/retrieve abc_0038_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89206/retrieve abc_0039_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89209/retrieve abc_0040_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89212/retrieve abc_0041_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89215/retrieve abc_0042_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89218/retrieve abc_0043_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89221/retrieve abc_0044_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89224/retrieve abc_0045_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89227/retrieve abc_0046_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89230/retrieve abc_0047_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89233/retrieve abc_0048_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89236/retrieve abc_0049_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89239/retrieve abc_0050_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89242/retrieve abc_0051_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89245/retrieve abc_0052_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89248/retrieve abc_0053_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89251/retrieve abc_0054_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89254/retrieve abc_0055_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89257/retrieve abc_0056_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89260/retrieve abc_0057_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89263/retrieve abc_0058_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89266/retrieve abc_0059_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89269/retrieve abc_0060_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89272/retrieve abc_0061_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89275/retrieve abc_0062_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89278/retrieve abc_0063_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89281/retrieve abc_0064_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89284/retrieve abc_0065_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89287/retrieve abc_0066_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89290/retrieve abc_0067_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89293/retrieve abc_0068_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89296/retrieve abc_0069_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89299/retrieve abc_0070_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89302/retrieve abc_0071_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89305/retrieve abc_0072_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89308/retrieve abc_0073_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89311/retrieve abc_0074_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89314/retrieve abc_0075_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89317/retrieve abc_0076_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89320/retrieve abc_0077_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89323/retrieve abc_0078_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89326/retrieve abc_0079_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89329/retrieve abc_0080_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89332/retrieve abc_0081_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89335/retrieve abc_0082_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89338/retrieve abc_0083_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89341/retrieve abc_0084_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89344/retrieve abc_0085_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89347/retrieve abc_0086_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89350/retrieve abc_0087_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89353/retrieve abc_0088_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89358/retrieve abc_0089_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89361/retrieve abc_0090_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89364/retrieve abc_0091_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89367/retrieve abc_0092_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89370/retrieve abc_0093_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89373/retrieve abc_0094_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89376/retrieve abc_0095_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89379/retrieve abc_0096_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89382/retrieve abc_0097_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89385/retrieve abc_0098_obj_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89388/retrieve abc_0099_obj_v00.7z
