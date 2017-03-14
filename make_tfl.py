import sys
import os


def touch(fname):
    open(fname, 'a').close()
    os.utime(fname, None)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print('Error: not part parameter')
        print('You should use: $ program <part>')
    part = sys.argv[1]
    if part.isdigit():
        part = int(part)
        part = 'tfl{0:02d}'.format(part)
        if not os.path.exists(part):
            os.mkdir(part)

        readme_path = os.path.join(part, 'README.md')
        print('Creating {}.'.format(readme_path))
        if not os.path.exists(readme_path):
            touch(readme_path)

        main_py_path = os.path.join(part, 'main.py')
        print('Creating {}.'.format(main_py_path))
        if not os.path.exists(main_py_path):
            touch(main_py_path)
        print('Project {} created'.format(part))
    else:
        print('Error: part is not a number')
