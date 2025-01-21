from proj.config_file import get_config_root
from pathlib import Path
import gdb

gdb.execute(f'directory {get_config_root(Path.cwd())}')
gdb.prompt_hook = lambda x: '(ffdb) '
gdb.execute('set history save on')
