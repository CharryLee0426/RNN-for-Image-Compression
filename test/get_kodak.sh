# !/bin/bash
# get kodak dataset for testing the model from Internet

mkdir -p test/images

for i in {0..24..1}; do
    echo ${i}
    wget http://r0k.us/graphics/kodak/kodak/kodim${i}.png -O test/images/kodim${i}.png
done