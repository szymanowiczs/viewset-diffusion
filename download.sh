mkdir -p data/skins2
mkdir -p data/skins

for id in $(cat data/meta2/*.txt | sort | uniq); do
  out=data/skins2/$id.png
  if test -e "$out"; then continue; fi
  wget --user-agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36" -S -O - https://www.minecraftskins.com/skin/download/$id >$out
done

for id in $(cat data/meta/*.txt | sort | uniq); do
  out=data/skins/$id.png
  if test -e "$out"; then continue; fi
  link=$(wget --user-agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36" -S -O - "https://www.planetminecraft.com/skin/$id/" | sed -n -E 's#.*href="([^"]+)".*Download.*#\1#p')
  wget --user-agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36" -S -O - "https://www.planetminecraft.com/$link" >$out
done