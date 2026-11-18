from pathlib import Path

for folder in ["data/dataset/labels/train", "data/dataset/labels/val"]:
    for txt_file in Path(folder).glob("*.txt"):
        lines = txt_file.read_text(encoding="utf-8").splitlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                parts[0] = str(int(parts[0]) + 1)
                new_lines.append(" ".join(parts))
        txt_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

print("Tous les fichiers .txt ont été décalés de +1 !")
