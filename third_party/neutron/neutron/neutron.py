import argparse
import sys
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="The proton command utility for profiling scripts and pytest tests.", usage="""
    proton [options] script.py [script_args] [script_options]
    proton [options] pytest [pytest_args] [script_options]
    python -m triton.profiler.proton [options] script.py [script_args] [script_options]
""", formatter_class=argparse.RawTextHelpFormatter)
    args, target_args = parser.parse_known_args()
    return args, target_args


def execute_as_main(script, args):
    script_path = os.path.abspath(script)
    # Prepare a clean global environment
    clean_globals = {
        "__name__": "__main__",
        "__file__": script_path,
        "__builtins__": __builtins__,
        sys.__name__: sys,
    }

    original_argv = sys.argv
    sys.argv = [script] + args
    # Append the script's directory in case the script uses relative imports
    sys.path.append(os.path.dirname(script_path))

    # Execute in the isolated environment
    try:
        with open(script_path, 'rb') as file:
            code = compile(file.read(), script_path, 'exec')
        exec(code, clean_globals)
    except Exception as e:
        print(f"An error occurred while executing the script: {e}")
    finally:
        sys.argv = original_argv


def run_instrumenting(args, target_args):

    script = target_args[0]
    script_args = target_args[1:] if len(target_args) > 1 else []
    execute_as_main(script, script_args)

def main():
    args, target_args = parse_arguments()
    run_instrumenting(args, target_args)


if __name__ == "__main__":
    main()
