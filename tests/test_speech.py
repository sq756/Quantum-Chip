import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from speech import LANGUAGES, get_message


def test_language_messages():
    required = {'en', 'zh', 'de', 'fr', 'es', 'el'}
    assert required.issubset(LANGUAGES)
    for code in required:
        msg = get_message(code, 'epoch', epoch=1, total=2, val_loss=0.5)
        assert '1/2' in msg
