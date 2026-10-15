"""For the zip members with a bad CRC-32: read the raw stored bytes (compress_type 0), recompute CRC, decode fully with ffmpeg, count frames. Writes crc_check.json."""
import zipfile, struct, zlib, subprocess, json, os, tempfile, shutil
R = "dataset/benchmark/mabe22/raw"; O = "results/vision/benchmark_mabe22/inventory"
tmp = f"/localhome/{os.environ['USER']}/{os.environ['SLURM_JOB_ID']}/crc"; os.makedirs(tmp, exist_ok=True)
out = {}
for z, i in [("mouse_submission", "QO9ONQTPOYAI26MYHPHY"), ("beetle_userTrain", "eb0e794d7f6e703d6d7a"), ("beetle_submission", "5411da61501cc4d42ad4"), ("beetle_submission", "868aea506ef41a301f3b")]:
    p = f"{R}/{z}_videos_resized_224.zip"; zf = zipfile.ZipFile(p); info = zf.getinfo(f"{i}.avi")
    with open(p, "rb") as fh:
        fh.seek(info.header_offset); h = fh.read(30); nlen, elen = struct.unpack("<HH", h[26:30]); fh.seek(info.header_offset + 30 + nlen + elen)
        data = fh.read(info.file_size)
    f = f"{tmp}/{i}.avi"; open(f, "wb").write(data)
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-f", "null", "-"], capture_output=True, text=True)
    n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames,width,height", "-of", "json", f], capture_output=True, text=True).stdout
    out[f"{z}/{i}"] = dict(stored_size=info.file_size, bytes_read=len(data), crc_computed=zlib.crc32(data), crc_in_zip=info.CRC, decode_stderr=r.stderr[:300], decode_rc=r.returncode, ffprobe=json.loads(n) if n else None)
shutil.rmtree(tmp)
json.dump(out, open(f"{O}/crc_check.json", "w"), indent=1); print(json.dumps(out, indent=1))
