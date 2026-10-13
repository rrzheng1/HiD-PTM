from pathlib import Path
import sys
if Path(sys.prefix).name.casefold()!='ptm_local':
    raise SystemExit('Please activate ptm_local first.')
print(f'Python: {sys.executable}')
