#!/bin/bash -e

IFS=$'\n' # new file name only after \n

if [  "$GITLAB_CI" == true ]; then
    files=($(git ls-files))
else
    files=($(git diff --cached --name-only --diff-filter=ACM))
    should_git_add=true
fi

for file in "${files[@]}" ; do
    if [[ $file == *.ipynb ]] ;
    then
        nb_dir=$(dirname "$file")
        if [[ $nb_dir == "." ]]; then
            nb_dir=""
        fi

        filename=$(basename "$file")
        stripped_dir=notebooks/stripped/${nb_dir} # copy the directory structure
        mkdir -p "$stripped_dir"
        target_stripped_file="${stripped_dir}/${filename%.ipynb}_stripped.py"

        python3 <(cat <<SCRIPT
import json
import sys

data = json.load(sys.stdin)
for cell in data['cells']:
    cell_type = cell['cell_type']
    source = cell['source']
    if cell_type == 'markdown':
        print('"""')
        for line in source:
            print(line.rstrip('\n'))
        print('"""\n')
    elif cell_type == 'code':
        for line in source:
            print(line.rstrip('\n'))
        print('\n')
    else:
        raise ValueError('Cell type', cell_type)
SCRIPT
        ) <"$file" >"$target_stripped_file"
		
        if [  "$should_git_add" == true ]; then 
          git add "$target_stripped_file" 
        fi

    fi
done
