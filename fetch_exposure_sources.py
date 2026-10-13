"""Download and hash documented occupational sources; keep the challenge CSV intact."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import urllib.request
import urllib.parse
import zipfile
import io

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'external'
MANIFEST = RAW / 'source_manifest.json'


def request(url):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Hackathon-occupational-source-audit'})
            with urllib.request.urlopen(req, timeout=40) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2)


def fetch_sources():
    RAW.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
        for item in manifest['files']:
            assert hashlib.sha256((RAW / item['file']).read_bytes()).hexdigest() == item['sha256']
        return manifest
    commit = json.loads(request('https://api.github.com/repos/AIOE-Data/AIOE/commits/main'))['sha']
    files = []

    def save(name, content, url, **metadata):
        (RAW / name).write_bytes(content)
        files.append({'file': name, 'url': url, 'bytes': len(content),
                      'sha256': hashlib.sha256(content).hexdigest(), **metadata})
        print(f'Saved {name}: {len(content):,} bytes', flush=True)

    for name in ['AIOE_DataAppendix.xlsx', 'Language Modeling AIOE and AIIE.xlsx',
                 'README.md', 'Generative AI/Generative AI Master.do']:
        url = f'https://raw.githubusercontent.com/AIOE-Data/AIOE/{commit}/' + urllib.parse.quote(name)
        save(Path(name).name, request(url), url, repository_commit=commit)
    url = 'https://www.onetcenter.org/dl_files/database/db_24_2_text.zip'
    archive_path = RAW / 'db_24_2_text.zip'
    archive = archive_path.read_bytes() if archive_path.exists() and zipfile.is_zipfile(archive_path) else request(url)
    save('db_24_2_text.zip', archive, url, version='24.2 (February 2020)',
         purpose='Historical 2010-SOC taxonomy compatible with the AIOE source; not current occupational estimates.')
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        for name in ['Skills.txt', 'Occupation Data.txt', 'Task Statements.txt', 'Alternate Titles.txt']:
            matches = [entry for entry in z.namelist() if entry.rsplit('/', 1)[-1] == name]
            assert len(matches) == 1, (name, matches)
            save(name, z.read(matches[0]), url, archive_member=matches[0], version='24.2')
    manifest = {'retrieved_at_utc': datetime.now(timezone.utc).isoformat(), 'files': files,
                'onet_attribution': 'O*NET 24.2 Database, USDOL/ETA, CC BY 4.0. Local derived tables aggregate, join and score source data; USDOL/ETA has not endorsed these modifications.',
                'onet_license_url': 'https://www.onetcenter.org/license_db.html',
                'aioe_citation': 'Felten, Raj and Seamans (2021), Strategic Management Journal 42(12):2195–2217; language-model extension (2023), https://arxiv.org/abs/2303.01157',
                'aioe_license_note': 'Authors request citation; no explicit redistribution license found in the downloaded repository README. Review source terms before public redistribution.',
                'target': 'Constructed language-model occupational exposure score, not observed automation or worker job-loss outcomes.'}
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    fetch_sources()
