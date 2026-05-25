"""Task 1: Data Handling & Memory Management — runs standalone."""
import os, sys, zipfile, time, platform
import pandas as pd
import numpy as np
import psutil

print('=== Task 1: Data Handling & Memory Management ===\n')

# ── Hardware & software info ─────────────────────────────────────────────────
print('=== Hardware & Software Setup ===')
print(f'OS            : {platform.system()} {platform.release()}')
print(f'Python        : {platform.python_version()}')
print(f'Pandas        : {pd.__version__}')
print(f'NumPy         : {np.__version__}')
mem = psutil.virtual_memory()
print(f'Total RAM     : {mem.total / 1e9:.1f} GB')
print(f'Available RAM : {mem.available / 1e9:.1f} GB')
disk = psutil.disk_usage('/')
print(f'Free disk     : {disk.free / 1e9:.1f} GB')
print(f'CPU (logical) : {psutil.cpu_count(logical=True)} cores')

# ── Dataset inventory ────────────────────────────────────────────────────────
DATASET_DIR = 'dataset'
ZIP_FILES = sorted([
    os.path.join(DATASET_DIR, f)
    for f in os.listdir(DATASET_DIR) if f.endswith('.zip')
])

total_zip_bytes, total_raw_bytes, all_members = 0, 0, []
for zpath in ZIP_FILES:
    total_zip_bytes += os.path.getsize(zpath)
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            if info.filename.endswith('.txt'):
                total_raw_bytes += info.file_size
                all_members.append((zpath, info.filename, info.file_size))

all_members.sort(key=lambda x: x[1])

print(f'\n=== Dataset Inventory ===')
print(f'Zip files found  : {len(ZIP_FILES)}')
print(f'Daily .txt files : {len(all_members)}')
print(f'Total zip size   : {total_zip_bytes/1e9:.2f} GB')
print(f'Total raw size   : {total_raw_bytes/1e9:.2f} GB')
print(f'Date range       : {all_members[0][1]} → {all_members[-1][1]}')

# ── Memory helpers ───────────────────────────────────────────────────────────
def mem_used_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6

def df_mem_mb(df):
    return df.memory_usage(deep=True).sum() / 1e6

mem_before_load = mem_used_mb()
print(f'\nBaseline process memory: {mem_before_load:.1f} MB')

# ── Load all files directly from zips ────────────────────────────────────────
print('\n=== Loading files (streaming from zip, 3 columns only) ===')
t0 = time.time()
chunks = []
for i, (zip_path, member_name, _) in enumerate(all_members):
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as fobj:
            day_df = pd.read_csv(
                fobj, sep='\t', header=None, usecols=[0, 1, 7],
                names=['square_id', 'time_ms', 'internet'],
                dtype={'square_id': 'int32', 'time_ms': 'int64'},
                na_values=[''], low_memory=False
            )
    day_agg = (
        day_df
        .groupby(['square_id', 'time_ms'], sort=False)['internet']
        .sum().reset_index()
    )
    day_agg['internet'] = day_agg['internet'].astype('float32')
    chunks.append(day_agg)
    elapsed = time.time() - t0
    print(f'  [{i+1:02d}/{len(all_members)}] {member_name}  '
          f'rows={len(day_agg):,}  elapsed={elapsed:.0f}s', flush=True)

mem_after_load = mem_used_mb()
print(f'\nAll files read in {time.time()-t0:.1f}s')
print(f'Memory after chunks: {mem_after_load:.1f} MB')

# ── Concatenate & deduplicate ─────────────────────────────────────────────────
print('\n=== Concatenating & deduplicating ===')
t1 = time.time()
raw_df = pd.concat(chunks, ignore_index=True)
del chunks
print(f'Shape: {raw_df.shape}  ({time.time()-t1:.1f}s)')

raw_df['timestamp'] = pd.to_datetime(raw_df['time_ms'], unit='ms', utc=True)
raw_df = raw_df.drop(columns=['time_ms'])
raw_df = raw_df.groupby(['square_id', 'timestamp'], sort=False)['internet'].sum().reset_index()
raw_df['internet'] = raw_df['internet'].astype('float32')

print(f'After dedup: {len(raw_df):,} rows')
print(f'Unique areas: {raw_df.square_id.nunique()}')
print(f'Unique timestamps: {raw_df.timestamp.nunique()}')
print(f'DataFrame memory: {df_mem_mb(raw_df):.1f} MB')

# ── Pivot ─────────────────────────────────────────────────────────────────────
print('\n=== Pivoting to wide format ===')
t2 = time.time()
pivot = raw_df.pivot(index='timestamp', columns='square_id', values='internet')
del raw_df
pivot.sort_index(inplace=True)
pivot.columns.name = None

missing_pct = pivot.isna().sum().sum() / pivot.size * 100
print(f'Pivot shape      : {pivot.shape}')
print(f'Missing values   : {missing_pct:.2f}%')
print(f'Memory (float64) : {df_mem_mb(pivot):.1f} MB')

pivot = pivot.fillna(0.0).astype('float32')
mem_final = mem_used_mb()
print(f'Memory (float32) : {df_mem_mb(pivot):.1f} MB')
print(f'Process memory   : {mem_final:.1f} MB')
print(f'Pivot done in {time.time()-t2:.1f}s')

# ── Save as Parquet ───────────────────────────────────────────────────────────
out_dir = 'data/processed'
os.makedirs(out_dir, exist_ok=True)
parquet_path = os.path.join(out_dir, 'milan_traffic_pivot.parquet')

print(f'\n=== Saving to {parquet_path} ===')
t3 = time.time()
pivot.to_parquet(parquet_path, compression='snappy')
pq_size = os.path.getsize(parquet_path)
print(f'Parquet file size: {pq_size/1e6:.1f} MB  ({time.time()-t3:.1f}s)')

# ── Memory summary ────────────────────────────────────────────────────────────
print('\n=== Memory Optimisation Summary ===')
print(f'  Raw text (estimated)      : {total_raw_bytes/1e6:,.0f} MB')
print(f'  3-col selection (~62% cut): {total_raw_bytes/1e6*3/8:,.0f} MB (estimated)')
print(f'  Final pivot (float32)     : {df_mem_mb(pivot):.1f} MB')
print(f'  Parquet on disk           : {pq_size/1e6:.1f} MB')
print(f'  Process memory baseline   : {mem_before_load:.1f} MB')
print(f'  Process memory at peak    : {mem_after_load:.1f} MB')
print(f'  Process memory final      : {mem_final:.1f} MB')
print(f'\nDate range: {pivot.index[0]} → {pivot.index[-1]}')
print('\nTask 1 COMPLETE ✓')
