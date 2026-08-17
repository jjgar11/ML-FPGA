#!/usr/bin/env python
"""
Registra el board Ultra96-v2 en la instalación de hls4ml del env activo.

hls4ml no trae ultra96v2 de fábrica; el flujo VivadoAccelerator lo necesita
(convert_model.py --backend VivadoAccelerator usa board="ultra96v2"). Antes esto
era un parche manual al paquete instalado y se perdía al recrear el env. Este
script lo hace reproducible a partir de los archivos versionados en
dev/hls4ml/ultra96v2_board/.

Uso (con el env destino activo):
    python dev/hls4ml/register_ultra96_board.py

Es idempotente: se puede correr varias veces sin romper nada.
"""
import json
import os
import shutil

import hls4ml

SRC = os.path.join(os.path.dirname(__file__), "ultra96v2_board")


def main():
    hls_dir = os.path.dirname(hls4ml.__file__)

    # 1) Agregar la entrada del board a supported_boards.json
    boards_json = os.path.join(
        hls_dir, "backends", "vivado_accelerator", "supported_boards.json"
    )
    with open(boards_json) as f:
        boards = json.load(f)

    entry = json.load(open(os.path.join(SRC, "supported_boards_entry.json")))
    if "ultra96v2" in boards:
        print("[skip] ultra96v2 ya estaba en supported_boards.json")
    else:
        boards.update(entry)
        with open(boards_json, "w") as f:
            json.dump(boards, f, indent=4)
        print(f"[ok] ultra96v2 agregado a {boards_json}")

    # 2) Copiar los templates (tcl_scripts + python_drivers)
    tpl_dest = os.path.join(
        hls_dir, "templates", "vivado_accelerator", "ultra96v2"
    )
    for sub in ("tcl_scripts", "python_drivers"):
        os.makedirs(os.path.join(tpl_dest, sub), exist_ok=True)
        for fname in os.listdir(os.path.join(SRC, sub)):
            shutil.copy2(
                os.path.join(SRC, sub, fname),
                os.path.join(tpl_dest, sub, fname),
            )
    print(f"[ok] templates copiados a {tpl_dest}")
    print("[done] Ultra96-v2 registrado en hls4ml:", hls_dir)


if __name__ == "__main__":
    main()
