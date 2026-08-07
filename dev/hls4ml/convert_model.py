import argparse
import os
import torch
import hls4ml

from mlfpga.config import MODELS_ROOT, HLS4ML_ROOT
from mlfpga.models.registry import get_model_spec


def make_hls_config(model, input_shape, mode, backend, reuse_factor=1):
    # VivadoAccelerator shares the same config structure as Vivado.
    # config_from_pytorch_model passes input_shape to create_initial_config(),
    # which VivadoAcceleratorBackend doesn't accept — use "Vivado" for config generation.
    config_backend = "Vivado" if backend == "VivadoAccelerator" else backend
    cfg = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=input_shape,
        granularity="name",
        backend=config_backend,
    )

    # Precision policy
    if mode == "int8":
        model_prec = "ap_fixed<16,6>"
        w_prec = "ap_fixed<8,2>"
        r_prec = "ap_fixed<16,6>"
        a_prec = "ap_fixed<24,10>"
    elif mode == "int4":
        model_prec = "ap_fixed<16,6>"
        w_prec = "ap_fixed<4,1>"
        r_prec = "ap_fixed<12,4>"
        a_prec = "ap_fixed<20,8>"
    else:
        raise ValueError("mode must be int8 or int4")

    cfg["Model"]["Precision"] = model_prec
    cfg["Model"]["ReuseFactor"] = reuse_factor

    for _, layer_cfg in cfg["LayerName"].items():
        layer_cfg["Precision"] = {
            "result": r_prec,
            "weight": w_prec,
            "bias": r_prec,
            "accum": a_prec,
        }

    return cfg


def load_model(model_name):
    spec = get_model_spec(model_name)
    pth_path = os.path.join(MODELS_ROOT, spec.pth_filename)

    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model weights not found: {pth_path}")

    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)

    # face_mlp: detect architecture from saved weights to handle variable num_classes
    if model_name == "face_mlp":
        from mlfpga.models.face_mlp import FaceMLP
        n_components = state_dict["net.0.weight"].shape[1]
        num_classes  = state_dict["net.4.weight"].shape[0]
        model = FaceMLP(n_components=n_components, num_classes=num_classes)
    else:
        model = spec.float_cls()

    model.load_state_dict(state_dict)
    model.eval()
    return spec, model


def _inject_custom_axi_wrapper(out_dir, model, spec):
    """
    hls_model.write() overwrites myproject_axi.cpp/h with a template that only sends
    1 AXI-Stream beat regardless of N_OUT.  Replace it with an explicit loop that reads
    N_IN float32 beats, runs inference, and writes N_OUT beats with TLAST on the last.

    Uses ap_axiu<32,0,0,0> so TDATA/TLAST/TKEEP/TSTRB come out as proper separate
    AXI4-Stream signals (32-bit TDATA) matching axi_dma_0 — the hls4ml stock struct
    wrapper ({float; ap_uint<1> last}) packs last into TDATA itself (64-bit, no real
    TLAST pin), which cannot connect to the DMA at all.
    """
    import torch as _torch

    with _torch.no_grad():
        dummy = _torch.zeros(1, *spec.input_shape_for_hls4ml)
        out   = model(dummy)
    n_in  = int(dummy[0].numel())
    n_out = int(out[0].numel())

    fw_dir = os.path.join(out_dir, "firmware")

    header = f"""\
#ifndef MYPROJECT_AXI_H_
#define MYPROJECT_AXI_H_

#include "myproject.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"

static const unsigned N_IN  = {n_in};
static const unsigned N_OUT = {n_out};

void myproject_axi(hls::stream<ap_axiu<32,0,0,0>>& in_r,
                   hls::stream<ap_axiu<32,0,0,0>>& out_r);

#endif
"""

    body = """\
#include "myproject_axi.h"
#include "myproject.h"

union f32_bits { float f; unsigned int u; };

static ap_uint<32> float_to_bits(float f) {
    f32_bits x; x.f = f; return (ap_uint<32>)x.u;
}
static float bits_to_float(ap_uint<32> b) {
    f32_bits x; x.u = (unsigned int)b; return x.f;
}

void myproject_axi(hls::stream<ap_axiu<32,0,0,0>>& in_r,
                   hls::stream<ap_axiu<32,0,0,0>>& out_r) {
#pragma HLS INTERFACE axis port=in_r
#pragma HLS INTERFACE axis port=out_r
#pragma HLS INTERFACE ap_ctrl_none port=return

    while (1) {
        input_t  in_buf[N_IN];
        result_t out_buf[N_OUT];

        for (unsigned i = 0; i < N_IN; i++) {
#pragma HLS PIPELINE
            ap_axiu<32,0,0,0> pkt = in_r.read();
            in_buf[i] = (input_t)bits_to_float(pkt.data);
        }

        myproject(in_buf, out_buf);

        for (unsigned j = 0; j < N_OUT; j++) {
#pragma HLS PIPELINE
            ap_axiu<32,0,0,0> pkt;
            pkt.data = float_to_bits((float)out_buf[j]);
            pkt.last = (j == N_OUT - 1) ? 1 : 0;
            pkt.keep = 0xF;
            pkt.strb = 0xF;
            out_r.write(pkt);
        }
    }
}
"""

    with open(os.path.join(fw_dir, "myproject_axi.h"), "w") as f:
        f.write(header)
    with open(os.path.join(fw_dir, "myproject_axi.cpp"), "w") as f:
        f.write(body)
    print(f"[INFO] Injected custom myproject_axi wrapper (N_IN={n_in}, N_OUT={n_out})")

    # Patch design.tcl: cap parallel jobs, route S2MM to HPC1 (separate from MM2S on HPC0),
    # and force c_s2mm_burst_size=1 to work around a DataMover bug where multi-beat AXI-M
    # bursts (AWLEN>0) only commit the first 4 bytes to DDR despite reporting IOC success.
    design_tcl = os.path.join(out_dir, "design.tcl")
    if os.path.exists(design_tcl):
        with open(design_tcl) as f:
            tcl = f.read()

        patched = tcl.replace("-jobs 6", "-jobs 2").replace("-jobs 4", "-jobs 2")

        # Enable HPC1_FPD (PSU GP1) at 32-bit alongside HPC0 (PSU GP0).
        patched = patched.replace(
            "CONFIG.PSU__SAXIGP0__DATA_WIDTH {32} \\\n]",
            "CONFIG.PSU__SAXIGP0__DATA_WIDTH {32} \\\n"
            "    CONFIG.PSU__USE__S_AXI_GP1 {1} \\\n"
            "    CONFIG.PSU__SAXIGP1__DATA_WIDTH {32} \\\n]",
        )

        # Redirect S2MM from shared HPC0 SmartConnect to dedicated HPC1 SmartConnect.
        patched = patched.replace(
            "Slave {/zynq_ultra_ps_e_0/S_AXI_HPC0_FPD} ddr_seg {Auto} intc_ip {/axi_smc}",
            "Slave {/zynq_ultra_ps_e_0/S_AXI_HPC1_FPD} ddr_seg {Auto} intc_ip {New AXI SmartConnect}",
        ).replace(
            "[get_bd_intf_pins axi_dma_0/M_AXI_S2MM]",
            "[get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC1_FPD]",
        )

        # Force S2MM burst size to 1 (AWLEN=0 per beat) — multi-beat bursts (AWLEN>0)
        # only write the first beat to DDR despite reporting success.
        patched = patched.replace(
            "CONFIG.c_s2mm_burst_size {256}",
            "CONFIG.c_s2mm_burst_size {2}",
        )

        if patched != tcl:
            with open(design_tcl, "w") as f:
                f.write(patched)
            print("[INFO] Patched design.tcl: -jobs → 2, S2MM → HPC1_FPD, burst_size → 1")


def convert(model_name, mode, part, backend, io_type_arg, reuse_factor, build, csim, synth):
    spec, model = load_model(model_name)

    io_type = io_type_arg if io_type_arg else (
        "io_parallel" if backend == "VivadoAccelerator" else spec.default_io_type
    )

    # 3) Build hls4ml config FROM PYTORCH (no ONNX)
    cfg = make_hls_config(
        model=model,
        input_shape=spec.input_shape_for_hls4ml,
        mode=mode,
        backend=backend,
        reuse_factor=reuse_factor,
    )

    # 4) Convert directly from PyTorch
    out_dir = os.path.join(
        HLS4ML_ROOT,
        f"{model_name}_hls4ml_pytorch_{mode}_{backend.lower()}",
    )

    extra_kwargs = {}
    if backend == "VivadoAccelerator":
        # ultra96v2 is registered in supported_boards.json with the correct part and TCL template.
        # Passing both board and part would conflict — board takes precedence in hls4ml.
        extra_kwargs["board"] = "ultra96v2"
    else:
        extra_kwargs["part"] = part

    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        hls_config=cfg,
        output_dir=out_dir,
        backend=backend,
        io_type=io_type,
        **extra_kwargs,
    )

    # 5) Write project files to disk (compile() builds C-sim .so, optional for synthesis)
    hls_model.write()
    if backend == "VivadoAccelerator":
        _inject_custom_axi_wrapper(out_dir, model, spec)
    else:
        hls_model.compile()

    print(f"[OK] hls4ml project generated at: {out_dir}")
    if backend == "VivadoAccelerator":
        print("[INFO] VivadoAccelerator: io_stream + AXI-DMA. Run synthesis in the VM via TCL scripts.")

    if build:
        # vsynth=True runs Vivado synthesis on top of HLS — produces bitstream for VivadoAccelerator
        hls_model.build(
            csim=csim,
            synth=synth,
            vsynth=(backend == "VivadoAccelerator"),
        )
        print("[OK] hls4ml build finished")
    else:
        print("[INFO] Build skipped. Use --build inside the VM where Vivado/Vitis is installed.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["mnist", "wine", "gtsrb", "gtsrb_gap", "gtsrb_gap_s", "nav_cnn", "face_mlp"],
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["int8", "int4"],
    )

    parser.add_argument(
        "--part",
        default="xczu3eg-sbva484-1-i",  # Ultra96-v2 exact part (speed grade -1-i)
    )

    parser.add_argument(
        "--backend",
        choices=["Vivado", "Vitis", "VivadoAccelerator"],
        default="Vivado",
    )

    parser.add_argument(
        "--io-type",
        choices=["io_parallel", "io_stream"],
        default=None,
        dest="io_type",
        help="Override io_type (default: io_parallel for Vivado, io_stream for VivadoAccelerator)",
    )

    parser.add_argument(
        "--reuse",
        type=int,
        default=1,
        dest="reuse_factor",
        help="ReuseFactor: higher = less parallel hardware, less RAM during synthesis (default: 1)",
    )

    parser.add_argument(
        "--build",
        action="store_true",
    )

    parser.add_argument(
        "--csim",
        action="store_true",
    )

    parser.add_argument(
        "--synth",
        action="store_true",
    )

    args = parser.parse_args()

    convert(
        model_name=args.model,
        mode=args.mode,
        part=args.part,
        backend=args.backend,
        io_type_arg=args.io_type,
        reuse_factor=args.reuse_factor,
        build=args.build,
        csim=args.csim,
        synth=args.synth,
    )


if __name__ == "__main__":
    main()
