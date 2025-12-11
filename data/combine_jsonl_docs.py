import json
import os

# folderul unde ai toate bmw_techworks_page_*.jsonl
input_folder = r"C:\Users\ionu_\PycharmProjects\BMWTechWorks_RAG_App\data"

# scriem rezultatul tot în folderul data
output_file = os.path.join(input_folder, "bmw_employees.jsonl")

seen_urls = set()
entries = []

for file in sorted(os.listdir(input_folder)):
    if file.startswith("bmw_techworks_page_") and file.endswith(".jsonl"):
        file_path = os.path.join(input_folder, file)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line.strip())
                if data["profile_url"] in seen_urls:
                    continue  # dedupe
                seen_urls.add(data["profile_url"])
                entries.append(data)

with open(output_file, "w", encoding="utf-8") as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

print(f"GATA! {len(entries)} angajați unici salvați în {output_file}")
