import os, json
from pathlib import Path
from collections import defaultdict

root = Path('Data')
inventory = defaultdict(lambda: {'count': 0, 'total_size_mb': 0, 'files': []})
folder_stats = defaultdict(lambda: {'count': 0, 'total_size_mb': 0, 'file_types': defaultdict(int)})

for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d != '.git']
    for f in filenames:
        fp = Path(dirpath) / f
        try:
            size_mb = fp.stat().st_size / (1024*1024)
        except:
            size_mb = 0
        ext = fp.suffix.lower()
        rel_folder = str(Path(dirpath).relative_to(root))

        inventory[ext]['count'] += 1
        inventory[ext]['total_size_mb'] += size_mb
        inventory[ext]['files'].append({'path': str(fp), 'size_mb': round(size_mb, 2), 'folder': rel_folder})

        top_folder = str(fp.relative_to(root)).split(os.sep)[0]
        folder_stats[top_folder]['count'] += 1
        folder_stats[top_folder]['total_size_mb'] += size_mb
        folder_stats[top_folder]['file_types'][ext] += 1

print('=== FILE INVENTORY BY EXTENSION ===')
for ext, data in sorted(inventory.items()):
    print(f'{ext:10s}: {data["count"]:3d} files, {data["total_size_mb"]:8.2f} MB')

print()
print('=== FOLDER STATS ===')
for folder, data in sorted(folder_stats.items()):
    types = dict(data['file_types'])
    print(f'{folder:25s}: {data["count"]:3d} files, {data["total_size_mb"]:8.2f} MB | {types}')

inv_out = {}
for ext, data in inventory.items():
    inv_out[ext] = {'count': data['count'], 'total_size_mb': round(data['total_size_mb'], 2), 'files': data['files']}
with open('COMPLETE_FILE_INVENTORY.json', 'w', encoding='utf-8') as f:
    json.dump(inv_out, f, indent=2, ensure_ascii=False)
print('\nSaved COMPLETE_FILE_INVENTORY.json')
