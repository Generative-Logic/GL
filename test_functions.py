# Generative Logic: A deterministic reasoning and knowledge generation engine.
# Copyright (C) 2025 Generative Logic UG (haftungsbeschränkt)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# ------------------------------------------------------------------------------
#
# This software is also available under a commercial license. For details,
# see: https://generative-logic.com/license
#
# Contributions to this project must be made under the terms of the
# Contributor License Agreement (CLA). See the project's CONTRIBUTING.md file.

import os


def generate_unique_lines_file(file1_path, file2_path, output_filename="unique_lines_output.txt"):
    """
    Generates a third text file containing unique lines from two input files.

    The output file will contain:
    - First, lines from file1 that are not present in file2.
    - Then, lines from file2 that are not present in file1.

    The output file is saved in the same directory as file1.

    Parameters:
    - file1_path (str): Path to the first input text file.
    - file2_path (str): Path to the second input text file.
    - output_filename (str): Name of the output file. Defaults to 'unique_lines_output.txt'.

    Raises:
    - FileNotFoundError: If either of the input files does not exist.
    - IOError: If there's an error reading or writing the files.
    """
    # Check if both files exist
    if not os.path.isfile(file1_path):
        raise FileNotFoundError(f"The file {file1_path} does not exist.")
    if not os.path.isfile(file2_path):
        raise FileNotFoundError(f"The file {file2_path} does not exist.")

    try:
        # Read lines from the first file
        with open(file1_path, 'r', encoding='utf-8') as f1:
            lines1 = set(line.rstrip('\n') for line in f1)

        # Read lines from the second file
        with open(file2_path, 'r', encoding='utf-8') as f2:
            lines2 = set(line.rstrip('\n') for line in f2)

        # Find unique lines
        unique_to_file1 = lines1 - lines2
        unique_to_file2 = lines2 - lines1

        # Prepare the output content
        output_lines = []

        if unique_to_file1:
            output_lines.append("Lines in file1 not in file2:\n")
            output_lines.extend(line + '\n' for line in sorted(unique_to_file1))
            output_lines.append("\n")  # Add a separator
        else:
            output_lines.append("No unique lines in file1 compared to file2.\n\n")

        if unique_to_file2:
            output_lines.append("Lines in file2 not in file1:\n")
            output_lines.extend(line + '\n' for line in sorted(unique_to_file2))
        else:
            output_lines.append("No unique lines in file2 compared to file1.\n")

        # Determine the directory of the first file
        directory = os.path.dirname(os.path.abspath(file1_path))
        output_path = os.path.join(directory, output_filename)

        # Write the output to the new file
        with open(output_path, 'w', encoding='utf-8') as out_file:
            out_file.writelines(output_lines)

        print(f"Output file created at: {output_path}")

    except IOError as e:
        print(f"An error occurred while processing the files: {e}")


def compare_files():
    file101 = r"c:\bin\test\n101.txt"
    file108 = r"c:\bin\test\n108.txt"

    def read_file_lines(filepath):
        """
        Reads a file and returns a set of stripped, non-empty lines.
        """
        with open(filepath, "r") as file:
            return {line.strip() for line in file if line.strip()}

    # Read lines from both files
    lines101 = read_file_lines(file101)
    lines108 = read_file_lines(file108)

    # Determine unique lines in each file
    only_in_101 = lines101 - lines108
    only_in_108 = lines108 - lines101

    # Print unique lines for each file label
    print("101:")
    for line in only_in_101:
        print(line)

    print("108:")
    for line in only_in_108:
        print(line)


import re
import itertools
import ast


def group_and_sort(filename):
    """
    Reads a text file containing a Python list of strings.
    Sorts the strings lexicographically using a key function in which all
    substrings matching the pattern "it_(\\d+)_lev_\\d+_" are replaced with "x".
    Then groups the sorted strings by their transformed key and prints each group.

    Parameters:
        filename (str): The path to the text file.
    """
    # Read file content and strip any extraneous whitespace
    with open(filename, 'r') as file:
        content = file.read().strip()

    # Parse the file content as a Python list of strings
    try:
        items = ast.literal_eval(content)
        if not isinstance(items, list):
            raise ValueError("File content is not a list.")
    except Exception as e:
        print(f"Error parsing file: {e}")
        return

    # Ensure all items are strings (in case they aren't)
    items = [str(item) for item in items]

    # Compile the regex pattern for substitution
    pattern = re.compile(r'it_\d+_lev_\d+_\d+')

    # Define a transformation function that substitutes matching substrings with "x"
    def transform(s):
        return pattern.sub("x", s)

    # Sort the items lexicographically based on the transformed string
    sorted_items = sorted(items, key=transform)

    # Group items that share the same transformed key
    groups = {}
    for key, group in itertools.groupby(sorted_items, key=transform):
        groups[key] = list(group)

    # Output the groups
    for transformed_key in sorted(groups.keys()):
        print(f"Group key (transformed): {transformed_key}")
        for item in groups[transformed_key]:
            print(item)
        print()  # Blank line between groups

# Example usage:
# If your file 'data.txt' contains:
# ['apple', 'it_10_lev_2_banana', 'orange', 'it_3_lev_1_banana', 'it_2_lev_2_banana']
# You can call:
# group_and_sort('data.txt')





