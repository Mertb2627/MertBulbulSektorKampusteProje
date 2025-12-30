import random
import shutil
from pathlib import Path

random.seed(42)

RAW_DIR = Path("data/raw-img")     # Sende ham veri burada
OUT_DIR = Path("data/animals10")   # train/val/test buraya oluşturulacak

SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}
IMG_EXTS = {".jpg", ".jpeg", ".png"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR bulunamadı: {RAW_DIR}. data/raw-img var mı?")

    # Eğer daha önce oluştuysa temizle
    if OUT_DIR.exists():
        print(f"{OUT_DIR} zaten var. Silip yeniden oluşturuyorum...")
        shutil.rmtree(OUT_DIR)

    classes = [d.name for d in RAW_DIR.iterdir() if d.is_dir()]
    classes.sort()
    print("Sınıflar:", classes)

    # klasörleri oluştur
    for split in SPLIT:
        for cls in classes:
            (OUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    # dosyaları böl ve kopyala
    for cls in classes:
        files = [p for p in (RAW_DIR / cls).iterdir() if p.is_file() and is_image(p)]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * SPLIT["train"])
        n_val = int(n * SPLIT["val"])

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        def copy_list(lst, split_name):
            for p in lst:
                dst = OUT_DIR / split_name / cls / p.name
                shutil.copy2(p, dst)

        copy_list(train_files, "train")
        copy_list(val_files, "val")
        copy_list(test_files, "test")

        print(f"{cls}: total={n} train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    print("✅ Hazır! Yeni yapı:", OUT_DIR)

if __name__ == "__main__":
    main()
