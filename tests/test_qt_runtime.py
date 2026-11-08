import os
import stat
import sys
from pathlib import Path

import pytest

from qt_runtime import clear_hidden_flags


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS-specific file flag')
def test_clear_hidden_flags_restores_qt_visible_files(tmp_path: Path) -> None:
    hidden_flag = getattr(stat, 'UF_HIDDEN', 0)
    if not hidden_flag:
        pytest.skip('UF_HIDDEN is unavailable')

    plugin_directory = tmp_path / 'platforms'
    plugin_directory.mkdir()
    plugin = plugin_directory / 'libqcocoa.dylib'
    plugin.write_bytes(b'plugin')
    os.chflags(plugin_directory, hidden_flag)
    os.chflags(plugin, hidden_flag)

    changed = clear_hidden_flags(plugin_directory)

    assert changed == 2
    assert not os.stat(plugin_directory).st_flags & hidden_flag
    assert not os.stat(plugin).st_flags & hidden_flag
