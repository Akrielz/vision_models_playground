import os
import re


def create_import_shortcuts(dir_path: str):
    # Get a list with all the files in the directory
    files = os.listdir(dir_path)

    all_imports = ""
    # Iterate over all the files
    for file in files:
        # Skip files starting with "__"
        if file.startswith("__"):
            continue

        # Get the path of the file
        file_path = os.path.join(dir_path, file)

        # Read the contents of the file
        with open(file_path, 'r') as f:
            # Get the contents of the file
            contents = f.read()

        # Identify all class name by the rule: class <class_name>(<parent_class_name>):
        # or by the rule: class <class_name>:
        class_names = re.findall(r'class\s+([a-zA-Z0-9_]+)[\s\(:]', contents)

        # Iterate over all the class names
        for class_name in class_names:
            if class_name.startswith("_"):
                continue

            # Prepare to import the class name
            import_string = "from " + file_path.replace("/", ".")[:-3] + " import " + class_name + "\n"

            # Append the import statement to the all imports string
            all_imports += import_string

    print(all_imports)
    # Write the all imports string to the file "__init__.py"
    with open(os.path.join(dir_path, "__init__.py"), 'w') as f:
        f.write(all_imports)


if __name__ == "__main__":
    create_import_shortcuts(dir_path="components/convolutions")