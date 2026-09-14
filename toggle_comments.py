"""
toggle_comments.py

Privremeno uklanja '#' komentare iz .py i .ipynb fajlova pre kacenja na Git,
i vraca ih nazad kad zavrsis. Pravi pravi backup pre bilo kakve izmene, tako
da nista nije nepovratno.

NE DIRA:
- docstringove (opisi klasa/funkcija u trostrukim navodnicima) - to je dokumentacija, ne "objasnjavajuci" komentar
- markdown celije u notebook-ovima (samo code celije se ciste)
- sadrzaj stringova (npr. url = "http://... # ne izgleda kao komentar" ostaje netaknut)

UPOTREBA (pokreni iz glavnog foldera projekta, gde su .py i .ipynb fajlovi):

    python toggle_comments.py strip      -> pravi backup i ciscenje komentara
    python toggle_comments.py restore    -> vraca fajlove iz backupa (sa komentarima)

Tok rada:
    1) python toggle_comments.py strip
    2) git add . / git commit / git push
    3) python toggle_comments.py restore
"""

import sys
import os
import json
import shutil
import tokenize
import io

BACKUP_DIR = "_backup_sa_komentarima"

# Dopuni ovu listu ako imas jos .py fajlova u projektu
PY_FILES = [
    "drone_dataset.py",
    "train_utils.py",
    "models.py",
    "api.py",
    "test_api.py",
]

IPYNB_FILES = [
    "01_explore_dataset.ipynb",
    "02_data_pipeline.ipynb",
    "03_baseline_model.ipynb",
    "04_automl.ipynb",
    "05_evaluation.ipynb",
]


def strip_comments(source: str) -> str:
    """
    Uklanja '#' komentare iz Python koda koriscenjem tokenize modula
    (ne obicnog regex-a) - ovo garantuje da se ne pokvari kod ako se
    '#' slucajno pojavi unutar stringa, i da se docstringovi ne diraju,
    jer ih tokenize prepoznaje kao STRING token, ne COMMENT token.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        filtered = [t for t in tokens if t.type != tokenize.COMMENT]
        return tokenize.untokenize(filtered)
    except Exception:
        # Ako celija ima npr. Jupyter magic (%matplotlib, !pip install) koji
        # nije validan Python, tokenize moze da pukne - u tom slucaju ostavljamo
        # tu celiju/fajl netaknutim, radije nego da rizikujemo da nesto pokvarimo
        return source


def strip_py_file(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    with open(path, "w", encoding="utf-8") as f:
        f.write(strip_comments(source))


def strip_ipynb_file(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue  # markdown celije ostaju netaknute
        source = cell.get("source", [])
        joined = "".join(source) if isinstance(source, list) else source
        stripped = strip_comments(joined)
        cell["source"] = stripped.splitlines(keepends=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")


def do_strip() -> None:
    candidates = [f for f in PY_FILES + IPYNB_FILES if os.path.exists(f)]
    if not candidates:
        print("Nijedan poznat fajl nije pronadjen u ovom folderu.")
        print("Pokreni skriptu iz glavnog foldera projekta, ili dopuni PY_FILES/IPYNB_FILES listu.")
        return

    os.makedirs(BACKUP_DIR, exist_ok=True)

    for fname in candidates:
        shutil.copy2(fname, os.path.join(BACKUP_DIR, fname))
        if fname.endswith(".py"):
            strip_py_file(fname)
        else:
            strip_ipynb_file(fname)
        print(f"Ociscen: {fname}   (original u {BACKUP_DIR}/{fname})")

    print(f"\nGotovo - {len(candidates)} fajlova ociscenih od komentara.")
    print("Sada mozes: git add . / git commit / git push")
    print("Kad zavrsis kacenje, pokreni:  python toggle_comments.py restore")


def do_restore() -> None:
    if not os.path.isdir(BACKUP_DIR):
        print(f"Folder '{BACKUP_DIR}' ne postoji - nema sta da se vrati.")
        print("(da li si vec pokrenuo 'restore', ili nikad nisi pokrenuo 'strip'?)")
        return

    restored = 0
    for fname in os.listdir(BACKUP_DIR):
        shutil.copy2(os.path.join(BACKUP_DIR, fname), fname)
        restored += 1
        print(f"Vracen: {fname}")

    print(f"\nGotovo - vraceno {restored} originalnih fajlova (sa komentarima).")
    print(f"Backup folder '{BACKUP_DIR}' je i dalje tu ako zatreba ponovo - obrisi ga rucno kad si siguran da ne treba.")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("strip", "restore"):
        print("Upotreba:")
        print("  python toggle_comments.py strip")
        print("  python toggle_comments.py restore")
        sys.exit(1)

    if sys.argv[1] == "strip":
        do_strip()
    else:
        do_restore()
