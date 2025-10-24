import os, re, time, logging, pandas as pd, openpyxl, requests, gdown
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('download_log_fixed.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

def get_links(file, sheet, col):
    log.info(f"Loading Excel: {file}")
    df = pd.read_excel(file, sheet_name=sheet)
    wb = openpyxl.load_workbook(file)
    ws = wb[sheet]
    idx = list(df.columns).index(col) + 1
    urls = []
    for i in range(len(df)):
        cell = ws.cell(row=i+2, column=idx)
        try: u = cell.hyperlink.target if cell.hyperlink else None
        except: u = None
        urls.append(u)
    df['ACTUAL_URL'] = urls
    log.info("Hyperlink extraction done")
    return df

def clean_name(name):
    for c in '<>:"/\\|?*': name = name.replace(c,'')
    return name.rstrip('.')[:200].strip()

def get_file_id(url):
    if not url: return None
    for pat in [r'/file/d/([a-zA-Z0-9_-]+)', r'/d/([a-zA-Z0-9_-]+)', r'id=([a-zA-Z0-9_-]+)']:
        m = re.search(pat, url)
        if m: return m.group(1)
    return None

def req_download(fid, path):
    s = requests.Session()
    URL = "https://drive.google.com/uc?export=download"
    r = s.get(URL, params={'id': fid}, stream=True)
    token = None
    for k,v in r.cookies.items():
        if k.startswith('download_warning'): token = v
    if not token:
        for l in r.text.split('\n'):
            if 'download_warning' in l or 'confirm=' in l:
                m = re.search(r'confirm=([^&"]+)', l)
                if m: token = m.group(1)
    params = {'id': fid}
    if token: params['confirm']=token
    r = s.get(URL, params=params, stream=True)
    with open(path,'wb') as f:
        for c in r.iter_content(32768):
            if c: f.write(c)
    return r.status_code==200

def download_multi(url, out, row):
    fid = get_file_id(url)
    if not fid: raise Exception("No file ID")
    for method in ['gdown_fuzzy','gdown_id','req','wget']:
        try:
            if method=='gdown_fuzzy':
                gdown.download(url, out, quiet=True, fuzzy=True)
            elif method=='gdown_id':
                gdown.download(id=fid, output=out, quiet=True)
            elif method=='req':
                if req_download(fid, out): return True
            else:
                r = requests.get(f"https://drive.google.com/uc?export=download&id={fid}&confirm=t", stream=True)
                with open(out,'wb') as f:
                    for c in r.iter_content(32768):
                        if c: f.write(c)
            if os.path.exists(out) and os.path.getsize(out)>0: return True
        except Exception as e:
            log.debug(f"Row {row}: {method} failed - {str(e)[:50]}")
    return False

try:
    log.info("="*60)
    log.info("STARTING DOWNLOADS")
    log.info("="*60)

    df = get_links('dataset.xlsx','Sheet1','LINK')
    out_dir = 'downloaded_audio_fixed'
    tmp_dir = 'temp'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    succ, fail = 0, 0
    fmt_stats, fail_det = {}, []

    for idx,row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        rnum = idx+2
        url = row['ACTUAL_URL']
        trans = row['TRANSCRIPT']

        if not url or pd.isna(url):
            msg = f"Row {rnum}: No URL"
            tqdm.write(f"❌ {msg}"); log.error(msg)
            fail+=1; fail_det.append((rnum, trans,'No URL')); continue
        if not trans or pd.isna(trans):
            msg = f"Row {rnum}: No transcript"
            tqdm.write(f"❌ {msg}"); log.error(msg)
            fail+=1; fail_det.append((rnum, url[:50],'No transcript')); continue

        tmp_path = os.path.join(tmp_dir, f"temp_{rnum}")
        try:
            if not download_multi(url,tmp_path,rnum): raise Exception("All download methods failed")
            _, ext = os.path.splitext(tmp_path)
            if not ext: ext='.bin'; log.warning(f"Row {rnum}: No ext, using .bin")
            fmt_stats[ext] = fmt_stats.get(ext,0)+1
            base = clean_name(trans); fname=base+ext; out_path=os.path.join(out_dir,fname)
            c=1
            while os.path.exists(out_path):
                fname=f"{base}_{c}{ext}"; out_path=os.path.join(out_dir,fname); c+=1
            os.rename(tmp_path,out_path)
            tqdm.write(f"✅ Row {rnum}: Downloaded {ext[1:].upper()} - {fname[:50]}")
            log.info(f"Row {rnum}: Downloaded {fname}")
            succ+=1; time.sleep(0.5)
        except Exception as e:
            tqdm.write(f"❌ Row {rnum}: FAILED - {str(e)[:100]}")
            log.error(f"Row {rnum}: {str(e)}"); log.error(f"Row {rnum}: URL - {url}")
            fail+=1; fail_det.append((rnum, trans[:50] if trans else 'N/A', str(e)[:100]))
            if os.path.exists(tmp_path): os.remove(tmp_path)
            continue

    log.info("="*60)
    log.info("DOWNLOAD COMPLETE")
    log.info(f"✅ Success: {succ} | ❌ Fail: {fail} | Rate: {(succ/(succ+fail)*100):.2f}%")
    log.info("="*60)

    log.info("Format Stats:")
    for k,v in sorted(fmt_stats.items(), key=lambda x:x[1], reverse=True):
        log.info(f" {k}: {v}")
    log.info("="*60)

    if fail_det:
        rpt='failed_downloads_fixed.txt'
        with open(rpt,'w',encoding='utf-8') as f:
            f.write(f"Failed Downloads Report\nGenerated: {datetime.now()}\nTotal Failed: {fail}\n{'='*60}\n\n")
            for rnum, ident, err in fail_det:
                f.write(f"Row {rnum}:\n  ID: {ident}\n  Error: {err}\n{'-'*60}\n")
        log.info(f"Failed report saved to '{rpt}'")
    log.info("Full log saved to 'download_log_fixed.txt'")

except Exception as e:
    log.critical(f"CRITICAL ERROR: {str(e)}", exc_info=True)
    raise
