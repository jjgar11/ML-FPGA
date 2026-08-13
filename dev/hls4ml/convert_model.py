import argparse
import os
import torch
import hls4ml

from mlfpga.config import MODELS_ROOT, HLS4ML_ROOT
from mlfpga.models.registry import get_model_spec


def _hls_input_shape(shape):
    """hls4ml expects the PER-SAMPLE input shape, without the batch dimension.

    Only normalize 4D conv inputs (1, C, H, W) -> (C, H, W). A leading batch dim
    on a conv input breaks hls4ml 1.2.0's VivadoAccelerator + io_parallel flow: the
    channels-last transpose optimizer miscomputes the shape ([3,64,64,1] ->
    [3,64,1]) and asserts. The registry is inconsistent here (gtsrb_gap carries the
    batch dim, gtsrb_gap_s does not), so we normalize instead of editing it — other
    consumers (dev/test_gtsrb.py) rely on the raw registry value.

    MLP shapes are left untouched: (13,) is a bare feature vector, and (1, 784) is
    required by models whose first layer is Flatten(start_dim=1) — dropping the
    leading dim there makes the input 1D and the flatten/transpose passes fail.
    """
    if len(shape) == 4 and shape[0] == 1:
        return tuple(shape[1:])
    return tuple(shape)


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
        dummy = _torch.zeros(1, *_hls_input_shape(spec.input_shape_for_hls4ml))
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
    print(f"[INFO] Injected custom myproject_axi wrapper (io_parallel, N_IN={n_in}, N_OUT={n_out})")
    _patch_design_tcl(out_dir)


def _streamify_axi_wrapper(out_dir):
    """
    io_stream variant (CNNs like gtsrb_gap). Here myproject() takes packed streams
    (hls::stream<nnet::array<...>>), so the io_parallel array wrapper does not apply.

    Instead of hand-writing the packing, transform hls4ml's OWN native wrapper: it
    already generates the correct enqueue/dequeue (flat -> input_t::size grouping),
    DATAFLOW and stream depths. We only swap its broken interface — the array of
    {float; ap_uint<1> last} structs (64-bit TDATA, no real TLAST) — for
    ap_axiu<32,0,0,0> ports with proper 32-bit TDATA + separate TLAST, doing the
    float<->bits conversion per beat. Everything else (packing order, depths) is
    left exactly as hls4ml produced it.
    """
    fw_dir = os.path.join(out_dir, "firmware")
    h_path = os.path.join(fw_dir, "myproject_axi.h")
    cpp_path = os.path.join(fw_dir, "myproject_axi.cpp")

    with open(h_path) as f:
        header = f.read()
    with open(cpp_path) as f:
        body = f.read()

    # --- header: add axis include + swap the prototype ---
    if '#include "ap_axi_sdata.h"' not in header:
        header = header.replace(
            '#include "myproject.h"',
            '#include "myproject.h"\n#include "ap_axi_sdata.h"\n#include "hls_stream.h"',
            1,
        )
    old_decl = "void myproject_axi(input_axi_t in[N_IN], output_axi_t out[N_OUT]);"
    new_decl = "void myproject_axi(hls::stream<ap_axiu<32,0,0,0>>& in_r, hls::stream<ap_axiu<32,0,0,0>>& out_r);"
    if old_decl not in header:
        raise RuntimeError(f"Unexpected native myproject_axi.h, cannot streamify: {h_path}")
    header = header.replace(old_decl, new_decl)

    # --- cpp: signature, pragmas, enqueue reads, dequeue writes ---
    old_sig = "void myproject_axi(input_axi_t in[N_IN], output_axi_t out[N_OUT]) {"
    new_sig = "void myproject_axi(hls::stream<ap_axiu<32,0,0,0>>& in_r, hls::stream<ap_axiu<32,0,0,0>>& out_r) {"
    old_pragmas = (
        "    #pragma HLS INTERFACE axis port=in\n"
        "    #pragma HLS INTERFACE axis port=out"
    )
    new_pragmas = (
        "    #pragma HLS INTERFACE axis port=in_r\n"
        "    #pragma HLS INTERFACE axis port=out_r"
    )
    old_enqueue = (
        "            ctype[j] = typename input_t::value_type(in[i * input_t::size + j].data);\n"
        "            is_last |= (in[i * input_t::size + j].last == 1)? true : false;"
    )
    new_enqueue = (
        "            ap_axiu<32,0,0,0> pkt = in_r.read();\n"
        "            ctype[j] = typename input_t::value_type(bits_to_float(pkt.data));\n"
        "            is_last |= (pkt.last == 1)? true : false;"
    )
    old_dequeue = "            out[i * result_t::size + j] = output_axi_t(ctype[j], last);"
    new_dequeue = (
        "            ap_axiu<32,0,0,0> pkt;\n"
        "            pkt.data = float_to_bits((float)ctype[j]);\n"
        "            pkt.last = last;\n"
        "            pkt.keep = 0xF;\n"
        "            pkt.strb = 0xF;\n"
        "            out_r.write(pkt);"
    )

    for needle in (old_sig, old_pragmas, old_enqueue, old_dequeue):
        if needle not in body:
            raise RuntimeError(f"Unexpected native myproject_axi.cpp, cannot streamify: {cpp_path}")

    helpers = (
        '#include "myproject_axi.h"\n'
        "\n"
        "union f32_bits { float f; unsigned int u; };\n"
        "static ap_uint<32> float_to_bits(float f) { f32_bits x; x.f = f; return (ap_uint<32>)x.u; }\n"
        "static float bits_to_float(ap_uint<32> b) { f32_bits x; x.u = (unsigned int)b; return x.f; }\n"
    )
    body = body.replace('#include "myproject_axi.h"', helpers, 1)
    body = body.replace(old_sig, new_sig)
    body = body.replace(old_pragmas, new_pragmas)
    body = body.replace(old_enqueue, new_enqueue)
    body = body.replace(old_dequeue, new_dequeue)

    with open(h_path, "w") as f:
        f.write(header)
    with open(cpp_path, "w") as f:
        f.write(body)
    print("[INFO] Streamified myproject_axi wrapper (io_stream): native packing kept, "
          "interface -> ap_axiu<32> in_r/out_r")
    _patch_design_tcl(out_dir)


def _patch_design_tcl(out_dir):
    """Patch design.tcl: cap parallel jobs, route S2MM to HPC1 (separate from MM2S on
    HPC0), and force c_s2mm_burst_size=1 to work around a DataMover bug where multi-beat
    AXI-M bursts (AWLEN>0) only commit the first 4 bytes to DDR despite reporting IOC
    success."""
    design_tcl = os.path.join(out_dir, "design.tcl")
    if not os.path.exists(design_tcl):
        return
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

    # Default io_type comes from the model spec (io_parallel for MLPs, io_stream for
    # CNNs). Both are supported for VivadoAccelerator: io_parallel -> array wrapper,
    # io_stream -> streamified wrapper. --io-type overrides per run.
    io_type = io_type_arg if io_type_arg else spec.default_io_type

    # 3) Build hls4ml config FROM PYTORCH (no ONNX)
    cfg = make_hls_config(
        model=model,
        input_shape=_hls_input_shape(spec.input_shape_for_hls4ml),
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
        # The AXI-Stream interface for the DMA depends on io_type:
        #   io_parallel -> myproject(input_t[], result_t[])   : array wrapper (MLPs, e.g. wine)
        #   io_stream   -> myproject(hls::stream<>&, ...)      : streamify native wrapper (CNNs, e.g. gtsrb_gap)
        if io_type == "io_stream":
            _streamify_axi_wrapper(out_dir)
        else:
            _inject_custom_axi_wrapper(out_dir, model, spec)
    else:
        hls_model.compile()

    print(f"[OK] hls4ml project generated at: {out_dir}")
    if backend == "VivadoAccelerator":
        print(f"[INFO] VivadoAccelerator: {io_type} + AXI-DMA. Run synthesis in the VM via TCL scripts.")

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
        help="Override io_type (default: the model spec's default_io_type — io_parallel for MLPs, io_stream for CNNs)",
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
