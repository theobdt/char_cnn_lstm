#!/bin/bash
mkdir -p ckpts
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}
gdrive_download 1-9bEAw5bx3htMiFW5uY_unLyCuKn9EBc ckpts/tmp.zip
unzip ckpts/tmp.zip -d ckpts/
rm ckpts/tmp.zip
