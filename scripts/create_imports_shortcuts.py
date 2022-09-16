import os
import re


def create_imports_for_all_sub_dirs(dir_path: str):
    # Walk in all the subdirectories
    for root, dirs, files in os.walk(dir_path):
        path = root.split(os.sep)

        # Check if the path contains a private directory
        private = False
        for path_component in path:
            if path_component.startswith("_"):
                private = True
                break

        if private:
            continue

        # Check if the current directory does not contain any public dir
        # Aka if it is leaf
        public_dirs = [dir for dir in dirs if not dir.startswith("_")]

        if len(public_dirs):
            continue

        # Create imports for the current directory
        create_imports_classes_and_functions(root)


def create_imports_classes_and_functions(dir_path: str):
    create_import_shortcuts(dir_path, keyword='class', append=False, description="Classes")
    create_import_shortcuts(dir_path, keyword='def', append=True, description="Functions")


def create_import_shortcuts(dir_path: str, keyword: str = 'class', append: bool = True, description: str = "Classes"):
    # Get a list with all the files in the directory
    files = os.listdir(dir_path)

    all_imports = f"# {description}\n"
    added = False

    # Iterate over all the files
    for file in files:
        # Skip files starting with "_"
        if file.startswith("_"):
            continue

        # Get the path of the file
        file_path = os.path.join(dir_path, file)

        # Read the contents of the file
        with open(file_path, 'r') as f:
            # Get the contents of the file
            contents = f.read()

        # Identify all keyword name by the rule: "keyword <class_name>(<parent_class_name>):"
        # or by the rule: "keyword <class_name>:", given that there is no spaces/tabs before the class keyword
        class_names = re.findall(keyword + r'\s+([a-zA-Z0-9_]+)', contents)

        # Iterate over all the class names
        for class_name in class_names:
            if class_name.startswith("_"):
                continue

            if class_name == "main":
                continue

            # check if there are tabs or spaces before the class keyword
            if contents[contents.find(f"{keyword} " + class_name) - 1] in "\t ":
                continue

            # Prepare to import the class name
            import_string = "from " + file_path.replace("/", ".")[:-3] + " import " + class_name + "\n"

            # Append the import statement to the all imports string
            all_imports += import_string

            added = True

    # Print the all imports string
    print(all_imports)

    # If not added any imports, then do not create a file
    if not added:
        return

    # Write the all imports string to the file "__init__.py"
    open_mode = "a" if append else "w"
    with open(os.path.join(dir_path, "__init__.py"), open_mode) as f:
        f.write(all_imports)


if __name__ == "__main__":
    create_imports_for_all_sub_dirs(dir_path="vision_models_playground/models/classifiers")
    create_imports_for_all_sub_dirs(dir_path="vision_models_playground/models/segmentation")
    create_imports_for_all_sub_dirs(dir_path="vision_models_playground/models/augmenters")
    create_imports_for_all_sub_dirs(dir_path="vision_models_playground/models/autoencoders")
    create_imports_for_all_sub_dirs(dir_path="vision_models_playground/components")
    create_imports_for_all_sub_dirs(dir_path="vision_models_playground/utility")
