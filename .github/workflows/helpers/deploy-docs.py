#! /usr/bin/env python3

import os
import tempfile
import subprocess
from typing import Tuple
from pathlib import Path
import shlex

KNOWN_HOSTS = '''
flexflow.ai ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAING3BmAIYc0G5hxIFPqQrgLjCt2t4vlRaLxds3QY6MaE
'''

def create_ssh_key_file(d: Path) -> Tuple[Path, Path]:
    d.chmod(0o700)

    ssh_private_key_path = d / 'id_ed25519'
    ssh_private_key_path.touch(mode=0o600, exist_ok=False)

    ssh_known_hosts_path = d / 'known_hosts'
    ssh_known_hosts_path.touch(mode=0o600, exist_ok=False)

    ssh_private_key_path.write_text(os.environ['SSH_PRIVATE_KEY'] + '\n')
    ssh_known_hosts_path.write_text(KNOWN_HOSTS)

    return ssh_private_key_path, ssh_known_hosts_path

def deploy(ssh_private_key_path: Path, ssh_known_hosts_path: Path) -> None:
    assert ssh_private_key_path.is_file()
    assert ssh_known_hosts_path.is_file()

    docs_html_dir = Path('./build/doxygen/html/')
    assert docs_html_dir.is_dir()
    assert (docs_html_dir / 'index.html').is_file()

    ssh_command = shlex.join([
        'ssh', '-i', str(ssh_private_key_path), '-o', f'UserKnownHostsFile={ssh_known_hosts_path}',
    ])
    print(ssh_command)
    subprocess.run(
        [
            'rsync',
            '--delete',
            '--recursive',
            '--verbose',
            '--human-readable',
            f'--rsh={ssh_command}',
            str(docs_html_dir) + '/',
            'deploy-ff-train-docs@flexflow.ai:/opt/www/ff-train-docs/',
        ],
        check=True,
    )

if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as _d:
        d = Path(_d)

        ssh_private_key_path, ssh_known_hosts_path = create_ssh_key_file(d)

        deploy(
            ssh_private_key_path=ssh_private_key_path,
            ssh_known_hosts_path=ssh_known_hosts_path,
        )
