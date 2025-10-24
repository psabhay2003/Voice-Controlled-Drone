import os, subprocess, csv
from datetime import datetime

def conv_bin_mp3(inp, out, ffmpeg, br='192k'):
    try:
        r = subprocess.run([ffmpeg, '-i', inp, '-y', '-acodec','libmp3lame','-ab',br,out],
                           capture_output=True, text=True, timeout=30)
        if r.returncode==0 and os.path.exists(out): return True, None
        return False, r.stderr[:200] if r.stderr else "Unknown error"
    except subprocess.TimeoutExpired: return False, "Timeout >30s"
    except Exception as e: return False, str(e)[:200]

def convert_all_bin(inp_dir='downloaded_audio_format_fixed', out_dir='downloaded_audio_fixed_to_mp3',
                    ffmpeg='ffmpeg.exe', br='192k'):
    
    if not os.path.exists(ffmpeg):
        print(f"FFmpeg not found at {ffmpeg}")
        return
    os.makedirs(out_dir, exist_ok=True)
    
    bins = [f for f in os.listdir(inp_dir) if f.endswith('.bin')]
    if not bins: print("No .bin files found"); return
    print(f"Found {len(bins)} files to convert\n"+"="*70)
    
    succ, fail, fail_det = 0,0,[]
    
    for i,f in enumerate(bins,1):
        inp_path = os.path.join(inp_dir,f)
        out_fname = f.replace('.bin','.mp3')
        out_path = os.path.join(out_dir,out_fname)
        print(f"[{i}/{len(bins)}] {f[:50]}...")
        ok, err = conv_bin_mp3(inp_path,out_path,ffmpeg,br)
        if ok: succ+=1; print(" Success")
        else: fail+=1; print(f" Failed: {err[:80]}"); fail_det.append({
            'filename':f,'error_reason':err,'file_path':inp_path,
            'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    print(f"Success: {succ} | Failed: {fail}")
    
    if fail_det:
        with open('failed_conversions.csv','w',newline='',encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['filename','error_reason','file_path','timestamp'])
            writer.writeheader(); writer.writerows(fail_det)
        print("Failed files saved to: failed_conversions.csv")


convert_all_bin()
