#!/bin/zsh
# usage: sr-full.sh <in-dir of 1920x1080 png> <out-dir>   (4x Real-ESRGAN per frame, scaled straight back to 1920x1080)
IN="$1"; OUT="$2"; B=/private/tmp/claude-501/-Users-dest1ny-Developer-Playground/fd1e34d6-1904-4a7c-b9ad-4623e8fa2150/scratchpad/sr/bin
mkdir -p "$OUT" "$OUT/.tmp"; n=0
for f in "$IN"/*.png; do b=$(basename "$f"); [ -f "$OUT/$b" ] && continue
  "$B/realesrgan-ncnn-vulkan" -i "$f" -o "$OUT/.tmp/$b" -n realesrgan-x4plus -s 4 -m "$B/models" >/dev/null 2>&1 || { echo "SR FAILED $b"; exit 1; }
  ffmpeg -v error -y -i "$OUT/.tmp/$b" -vf "scale=1920:1080:flags=lanczos" "$OUT/$b" && rm -f "$OUT/.tmp/$b"; n=$((n+1)); [ $((n % 10)) -eq 0 ] && echo "$(date +%H:%M:%S) $n done"
done; echo "SR-FULL DONE $(ls "$OUT"/*.png | wc -l) frames"
