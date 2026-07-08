import os
import pickle
from deepface import DeepFace

# ===== CHANGE THIS PATH IF NEEDED =====
images_path = r"D:\Family_Face_Recognization\images"

database = []

# Loop through each person's folder
for person_name in os.listdir(images_path):

    person_folder = os.path.join(images_path, person_name)

    # Skip files, process only folders
    if not os.path.isdir(person_folder):
        continue

    # Loop through all images of that person
    for image_name in os.listdir(person_folder):

        image_path = os.path.join(person_folder, image_name)

        try:
            # Generate embedding
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name="Facenet512",
                enforce_detection=False
            )[0]["embedding"]

            # Store name + embedding
            database.append({
                "name": person_name,
                "embedding": embedding
            })

            print(f"Processed: {person_name} -> {image_name}")

        except Exception as e:
            print(f"Skipped {image_name}: {e}")

# Save all embeddings
with open("models/encodings_v4.pkl", "wb") as file:
    pickle.dump(database, file)

print("\n===================================")
print("encodings_v4.pkl created successfully!")
print(f"Total embeddings stored: {len(database)}")
print("===================================")