#!/bin/bash

cat > stl2_v00.txt << EOF
https://archive.nyu.edu/rest/bitstreams/88599/retrieve abc_0000_stl2_v00.7z
https://archive.nyu.edu/rest/bitstreams/88603/retrieve abc_0001_stl2_v00.7z
https://archive.nyu.edu/rest/bitstreams/88635/retrieve abc_0002_stl2_v00.7z
https://archive.nyu.edu/rest/bitstreams/88639/retrieve abc_0003_stl2_v00.7z
https://archive.nyu.edu/rest/bitstreams/88643/retrieve abc_0004_stl2_v00.7z
https://archive.nyu.edu/rest/bitstreams/88647/retrieve abc_0005_stl2_v00.7z
https://archive.nyu.edu/rest/bitstreams/88651/retrieve abc_0006_stl2_v00.7z
https://archive.nyu.edu/rest/bitstreams/88655/retrieve abc_0007_stl2_v00.7z
EOF

cat stl2_v00.txt | xargs -n 2 -P 8 sh -c 'wget --no-check-certificate $0 -O data/ABC-Dataset/$1'
rm stl2_v00.txt

for file in ./*.7z; do
    filename=$(basename "$file" .7z)
    mkdir "$filename"
    7z x "$file" -o"$filename"
done

# more files
# https://archive.nyu.edu/rest/bitstreams/88659/retrieve abc_0008_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88663/retrieve abc_0009_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88667/retrieve abc_0010_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88671/retrieve abc_0011_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88675/retrieve abc_0012_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88679/retrieve abc_0013_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88683/retrieve abc_0014_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88687/retrieve abc_0015_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88691/retrieve abc_0016_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88695/retrieve abc_0017_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88699/retrieve abc_0018_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88703/retrieve abc_0019_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88708/retrieve abc_0020_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88712/retrieve abc_0021_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88716/retrieve abc_0022_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88720/retrieve abc_0023_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88724/retrieve abc_0024_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88728/retrieve abc_0025_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88732/retrieve abc_0026_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88736/retrieve abc_0027_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88740/retrieve abc_0028_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88744/retrieve abc_0029_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88748/retrieve abc_0030_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88752/retrieve abc_0031_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88756/retrieve abc_0032_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88760/retrieve abc_0033_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88764/retrieve abc_0034_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88768/retrieve abc_0035_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88772/retrieve abc_0036_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88776/retrieve abc_0037_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88780/retrieve abc_0038_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88784/retrieve abc_0039_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88788/retrieve abc_0040_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88792/retrieve abc_0041_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88796/retrieve abc_0042_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88800/retrieve abc_0043_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88804/retrieve abc_0044_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88808/retrieve abc_0045_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88812/retrieve abc_0046_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88817/retrieve abc_0047_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88821/retrieve abc_0048_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88825/retrieve abc_0049_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88831/retrieve abc_0050_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88835/retrieve abc_0051_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88839/retrieve abc_0052_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88843/retrieve abc_0053_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88847/retrieve abc_0054_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88851/retrieve abc_0055_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88855/retrieve abc_0056_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88859/retrieve abc_0057_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88870/retrieve abc_0058_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88874/retrieve abc_0059_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88878/retrieve abc_0060_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88882/retrieve abc_0061_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88886/retrieve abc_0062_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88890/retrieve abc_0063_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88894/retrieve abc_0064_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88898/retrieve abc_0065_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88902/retrieve abc_0066_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88906/retrieve abc_0067_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88910/retrieve abc_0068_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88914/retrieve abc_0069_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88918/retrieve abc_0070_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88922/retrieve abc_0071_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88926/retrieve abc_0072_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88930/retrieve abc_0073_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88934/retrieve abc_0074_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88938/retrieve abc_0075_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88942/retrieve abc_0076_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88946/retrieve abc_0077_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88987/retrieve abc_0078_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88991/retrieve abc_0079_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88995/retrieve abc_0080_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88999/retrieve abc_0081_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89003/retrieve abc_0082_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89007/retrieve abc_0083_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89011/retrieve abc_0084_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89015/retrieve abc_0085_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89019/retrieve abc_0086_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89023/retrieve abc_0087_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89027/retrieve abc_0088_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89031/retrieve abc_0089_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88951/retrieve abc_0090_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88955/retrieve abc_0091_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88959/retrieve abc_0092_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88963/retrieve abc_0093_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88967/retrieve abc_0094_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88971/retrieve abc_0095_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88975/retrieve abc_0096_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88979/retrieve abc_0097_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/88983/retrieve abc_0098_stl2_v00.7z
# https://archive.nyu.edu/rest/bitstreams/89035/retrieve abc_0099_stl2_v00.7z
