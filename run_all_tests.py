import os


def ejecutar_resultados():
    current_dir = os.getcwd()
    files = os.listdir(current_dir)

    py_files = [f for f in files if f.endswith('.py')]
    py_files = sorted(py_files)

    for py_file in py_files:
        if py_file != "run_all_tests.py":
            # print(f"Ejecutando {py_file}...")
            try:
                exec(open(py_file).read(), globals())
                print(f"------------")
            except Exception as e:
                print(f"Error executing {py_file}: {e}")


if __name__ == "__main__":
    ejecutar_resultados()
