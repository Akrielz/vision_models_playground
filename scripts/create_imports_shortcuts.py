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
        # Aka if it is a leaf
        public_dirs = [d for d in dirs if not d.startswith("_")]

        if len(public_dirs):
            continue

        # Create imports for the current directory
        create_imports_classes_and_functions(root)


def create_imports_classes_and_functions(dir_path: str):
    # Delete the previous __init__.py file
    init_path = os.path.join(dir_path, '__init__.py')
    if os.path.exists(init_path):
        os.remove(init_path)

    # Create the new __init__.py file
    all_classes = create_import_shortcuts(dir_path, keyword='class', append=False, description="# Classes")
    all_functions = create_import_shortcuts(dir_path, keyword='def', append=True, description="# Functions")

    if not os.path.exists(init_path):
        return

    all_keywords = all_classes + all_functions

    # Put all keywords in the __all__ list
    all_string = "\n# All imports \n__all__ = [\n"
    for keyword in all_keywords:
        all_string += f"    '{keyword}',\n"
    all_string += "]\n"

    # Write the __init__.py file
    with open(init_path, 'a') as f:
        f.write(all_string)


def create_import_shortcuts(
        dir_path: str,
        keyword: str = 'class',
        append: bool = True,
        description: str = "Classes"
):
    # Get a list with all the files in the directory
    files = os.listdir(dir_path)

    all_imports = f"{description}\n"
    added = False

    all_classes = []

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

        if file == "evaluate_models.py":
            print('hi')

        # Identify all keyword name by the rule: "keyword <class_name>(<parent_class_name>):"
        # or by the rule: "keyword <class_name>:", given that there is no spaces/tabs before the class keyword
        class_names = re.findall(keyword + r'\s+([a-zA-Z0-9_]+)', contents)

        # Get the positions of the class_names with regex
        class_names_positions = [m.start() for m in re.finditer(keyword + r'\s+([a-zA-Z0-9_]+)', contents)]

        # Iterate over all the class names
        for class_name, position in zip(class_names, class_names_positions):
            if class_name.startswith("_"):
                continue

            if class_name == "main":
                continue

            # check if there are tabs or spaces before the class keyword
            if contents[position - 1] in "\t ":
                continue

            # Prepare to import the class name
            import_string = "from " + file_path.replace("/", ".")[:-3] + " import " + class_name + "\n"

            # Append the import statement to the all imports string
            all_imports += import_string
            all_classes.append(class_name)

            added = True

    # Print the all imports string
    print(all_imports)

    # If not added any imports, then do not create a file
    if not added:
        return []

    # Check if the __init__.py file exists
    init_path = os.path.join(dir_path, '__init__.py')
    if os.path.exists(init_path):
        all_imports = "\n" + all_imports

    # Write the all imports string to the file "__init__.py"
    open_mode = "a" if append else "w"
    with open(init_path, open_mode) as f:
        f.write(all_imports)

    return all_classes


if __name__ == "__main__":
    create_imports_for_all_sub_dirs(
        dir_path="vision_models_playground/external"
    )
