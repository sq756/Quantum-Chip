LANGUAGES = {
    'en': {'name': 'English', 'epoch': 'Epoch {epoch}/{total} completed. Validation loss {val_loss:.4f}'},
    'zh': {'name': '中文', 'epoch': '第 {epoch}/{total} 轮完成。验证损失 {val_loss:.4f}'},
    'de': {'name': 'Deutsch', 'epoch': 'Epoche {epoch}/{total} abgeschlossen. Validierungsverlust {val_loss:.4f}'},
    'fr': {'name': 'Français', 'epoch': "Époque {epoch}/{total} terminée. Perte de validation {val_loss:.4f}"},
    'es': {'name': 'Español', 'epoch': 'Época {epoch}/{total} completada. Pérdida de validación {val_loss:.4f}'},
    'el': {'name': 'Ελληνικά', 'epoch': 'Εποχή {epoch}/{total} ολοκληρώθηκε. Απώλεια επικύρωσης {val_loss:.4f}'},
}


def get_message(lang: str, key: str, **kwargs) -> str:
    data = LANGUAGES.get(lang, LANGUAGES['en'])
    template = data.get(key, '')
    return template.format(**kwargs)


def speak_message(lang: str, key: str, **kwargs) -> None:
    text = get_message(lang, key, **kwargs)
    try:
        import pyttsx3
    except Exception:
        return
    engine = pyttsx3.init()
    try:
        for voice in engine.getProperty('voices'):
            langs = getattr(voice, 'languages', [])
            if any(lang in str(l).lower() for l in langs):
                engine.setProperty('voice', voice.id)
                break
    except Exception:
        pass
    engine.say(text)
    engine.runAndWait()
