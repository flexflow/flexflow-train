from proj import get_repo_root
from pathlib import Path
import gdb

gdb.execute(f'directory {get_repo_root(Path.cwd())}')
gdb.prompt_hook = lambda x: '(ffdb) '
gdb.execute('set history save on')
gdb.execute('catch throw')
