import os

def make_folder_structure():
    """
    Create a folder structure for the project.

    Args:
        base_path (str): The base path where the folder structure will be created.
    """
    folders = [
        "data/landing",
        "data/formatted",
        "data/trusted",
        "data/exploitaition"
    ]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
        else:
            print(f"Folder already exists: {folder}")
    print("Folder structure ready.")