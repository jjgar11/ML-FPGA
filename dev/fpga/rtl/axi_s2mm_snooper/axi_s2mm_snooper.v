`timescale 1ns / 1ps
//
// axi_s2mm_snooper.v
//
// Transparent AXI4 write-only passthrough inserted between DMA M_AXI_S2MM
// and the SmartConnect.  All signals are wired straight through (zero latency,
// zero logic in the data path).  A set of capture registers latches key bus
// events and exposes them via an AXI4-Lite slave so Linux can read them with
// a normal /dev/uio mapping.
//
// Register map (AXI-Lite, 32-bit reads):
//   0x00  AWADDR    — address of the last AW transaction
//   0x04  AWLEN     — burst length of the last AW transaction (AWLEN+1 beats)
//   0x08  WDATA0    — WDATA on beat 0 of the last burst
//   0x0C  WDATA1    — WDATA on beat 1
//   0x10  WDATA2    — WDATA on beat 2
//   0x14  WLAST_AT  — beat index (0-based) when WLAST fired
//   0x18  AW_CNT    — total AW transactions since last clear
//   0x1C  W_CNT     — total W beats since last clear
//   0x20  B_CNT     — total B responses since last clear
//   0x24  BRESP     — BRESP of the last B response (0=OKAY, 2=SLVERR, 3=DECERR)
//   0x28  CLEAR     — write 1 to reset all counters and captured values
//

module axi_s2mm_snooper (
    input  wire        aclk,
    input  wire        aresetn,

    // -----------------------------------------------------------------
    // AXI4 write-only SLAVE  (connects to DMA M_AXI_S2MM)
    // -----------------------------------------------------------------
    input  wire [31:0] S_AXI_AWADDR,
    input  wire [7:0]  S_AXI_AWLEN,
    input  wire [2:0]  S_AXI_AWSIZE,
    input  wire [1:0]  S_AXI_AWBURST,
    input  wire [0:0]  S_AXI_AWLOCK,
    input  wire [3:0]  S_AXI_AWCACHE,
    input  wire [2:0]  S_AXI_AWPROT,
    input  wire [3:0]  S_AXI_AWQOS,
    input  wire        S_AXI_AWVALID,
    output wire        S_AXI_AWREADY,

    input  wire [31:0] S_AXI_WDATA,
    input  wire [3:0]  S_AXI_WSTRB,
    input  wire        S_AXI_WLAST,
    input  wire        S_AXI_WVALID,
    output wire        S_AXI_WREADY,

    output wire [1:0]  S_AXI_BRESP,
    output wire        S_AXI_BVALID,
    input  wire        S_AXI_BREADY,

    // -----------------------------------------------------------------
    // AXI4 write-only MASTER  (connects to SmartConnect S00_AXI)
    // -----------------------------------------------------------------
    output wire [31:0] M_AXI_AWADDR,
    output wire [7:0]  M_AXI_AWLEN,
    output wire [2:0]  M_AXI_AWSIZE,
    output wire [1:0]  M_AXI_AWBURST,
    output wire [0:0]  M_AXI_AWLOCK,
    output wire [3:0]  M_AXI_AWCACHE,
    output wire [2:0]  M_AXI_AWPROT,
    output wire [3:0]  M_AXI_AWQOS,
    output wire        M_AXI_AWVALID,
    input  wire        M_AXI_AWREADY,

    output wire [31:0] M_AXI_WDATA,
    output wire [3:0]  M_AXI_WSTRB,
    output wire        M_AXI_WLAST,
    output wire        M_AXI_WVALID,
    input  wire        M_AXI_WREADY,

    input  wire [1:0]  M_AXI_BRESP,
    input  wire        M_AXI_BVALID,
    output wire        M_AXI_BREADY,

    // -----------------------------------------------------------------
    // AXI4-Lite SLAVE  (debug register readback, connected to PS)
    // -----------------------------------------------------------------
    input  wire [5:0]  S_AXI_LITE_ARADDR,
    input  wire        S_AXI_LITE_ARVALID,
    output reg         S_AXI_LITE_ARREADY,
    output reg  [31:0] S_AXI_LITE_RDATA,
    output reg  [1:0]  S_AXI_LITE_RRESP,
    output reg         S_AXI_LITE_RVALID,
    input  wire        S_AXI_LITE_RREADY,

    input  wire [5:0]  S_AXI_LITE_AWADDR,
    input  wire        S_AXI_LITE_AWVALID,
    output reg         S_AXI_LITE_AWREADY,
    input  wire [31:0] S_AXI_LITE_WDATA,
    input  wire [3:0]  S_AXI_LITE_WSTRB,
    input  wire        S_AXI_LITE_WVALID,
    output reg         S_AXI_LITE_WREADY,
    output reg  [1:0]  S_AXI_LITE_BRESP,
    output reg         S_AXI_LITE_BVALID,
    input  wire        S_AXI_LITE_BREADY
);

// =================================================================
// Transparent passthrough — purely combinational, zero latency
// =================================================================
assign M_AXI_AWADDR  = S_AXI_AWADDR;
assign M_AXI_AWLEN   = S_AXI_AWLEN;
assign M_AXI_AWSIZE  = S_AXI_AWSIZE;
assign M_AXI_AWBURST = S_AXI_AWBURST;
assign M_AXI_AWLOCK  = S_AXI_AWLOCK;
assign M_AXI_AWCACHE = S_AXI_AWCACHE;
assign M_AXI_AWPROT  = S_AXI_AWPROT;
assign M_AXI_AWQOS   = S_AXI_AWQOS;
assign M_AXI_AWVALID = S_AXI_AWVALID;
assign S_AXI_AWREADY = M_AXI_AWREADY;

assign M_AXI_WDATA  = S_AXI_WDATA;
assign M_AXI_WSTRB  = S_AXI_WSTRB;
assign M_AXI_WLAST  = S_AXI_WLAST;
assign M_AXI_WVALID = S_AXI_WVALID;
assign S_AXI_WREADY = M_AXI_WREADY;

assign S_AXI_BRESP  = M_AXI_BRESP;
assign S_AXI_BVALID = M_AXI_BVALID;
assign M_AXI_BREADY = S_AXI_BREADY;

// =================================================================
// Capture registers
// =================================================================
reg [31:0] cap_awaddr   = 0;
reg [7:0]  cap_awlen    = 0;
reg [31:0] cap_wdata0   = 0;
reg [31:0] cap_wdata1   = 0;
reg [31:0] cap_wdata2   = 0;
reg [7:0]  cap_wlast_at = 0;
reg [31:0] cap_aw_cnt   = 0;
reg [31:0] cap_w_cnt    = 0;
reg [31:0] cap_b_cnt    = 0;
reg [1:0]  cap_bresp    = 0;
reg [7:0]  beat_idx     = 0;
// Combinational clear pulse: high for the one cycle when PS writes 1 to 0x28
wire do_clear = S_AXI_LITE_AWVALID && S_AXI_LITE_WVALID && !S_AXI_LITE_BVALID &&
                (S_AXI_LITE_AWADDR[5:2] == 4'hA) && S_AXI_LITE_WDATA[0];

always @(posedge aclk) begin
    if (!aresetn || do_clear) begin
        cap_awaddr   <= 0; cap_awlen    <= 0;
        cap_wdata0   <= 0; cap_wdata1   <= 0; cap_wdata2 <= 0;
        cap_wlast_at <= 0;
        cap_aw_cnt   <= 0; cap_w_cnt    <= 0; cap_b_cnt  <= 0;
        cap_bresp    <= 0; beat_idx     <= 0;
    end else begin

        // AW handshake
        if (S_AXI_AWVALID && M_AXI_AWREADY) begin
            cap_awaddr <= S_AXI_AWADDR;
            cap_awlen  <= S_AXI_AWLEN;
            cap_aw_cnt <= cap_aw_cnt + 1;
            beat_idx   <= 0;
        end

        // W handshake — store first 3 beats and track WLAST position
        if (S_AXI_WVALID && M_AXI_WREADY) begin
            case (beat_idx)
                8'd0: cap_wdata0 <= S_AXI_WDATA;
                8'd1: cap_wdata1 <= S_AXI_WDATA;
                8'd2: cap_wdata2 <= S_AXI_WDATA;
                default: ;
            endcase
            if (S_AXI_WLAST) begin
                cap_wlast_at <= beat_idx;
                beat_idx     <= 0;
            end else begin
                beat_idx <= beat_idx + 1;
            end
            cap_w_cnt <= cap_w_cnt + 1;
        end

        // B handshake
        if (M_AXI_BVALID && S_AXI_BREADY) begin
            cap_bresp  <= M_AXI_BRESP;
            cap_b_cnt  <= cap_b_cnt + 1;
        end
    end
end

// =================================================================
// AXI4-Lite slave — read path
// =================================================================
always @(posedge aclk) begin
    if (!aresetn) begin
        S_AXI_LITE_ARREADY <= 0;
        S_AXI_LITE_RVALID  <= 0;
        S_AXI_LITE_RRESP   <= 0;
        S_AXI_LITE_RDATA   <= 0;
    end else begin
        S_AXI_LITE_ARREADY <= 0;

        if (S_AXI_LITE_ARVALID && !S_AXI_LITE_RVALID) begin
            S_AXI_LITE_ARREADY <= 1;
            S_AXI_LITE_RVALID  <= 1;
            S_AXI_LITE_RRESP   <= 2'b00;
            case (S_AXI_LITE_ARADDR[5:2])
                4'h0: S_AXI_LITE_RDATA <= cap_awaddr;
                4'h1: S_AXI_LITE_RDATA <= {24'h0, cap_awlen};
                4'h2: S_AXI_LITE_RDATA <= cap_wdata0;
                4'h3: S_AXI_LITE_RDATA <= cap_wdata1;
                4'h4: S_AXI_LITE_RDATA <= cap_wdata2;
                4'h5: S_AXI_LITE_RDATA <= {24'h0, cap_wlast_at};
                4'h6: S_AXI_LITE_RDATA <= cap_aw_cnt;
                4'h7: S_AXI_LITE_RDATA <= cap_w_cnt;
                4'h8: S_AXI_LITE_RDATA <= cap_b_cnt;
                4'h9: S_AXI_LITE_RDATA <= {30'h0, cap_bresp};
                default: S_AXI_LITE_RDATA <= 32'hDEAD_BEEF;
            endcase
        end

        if (S_AXI_LITE_RVALID && S_AXI_LITE_RREADY)
            S_AXI_LITE_RVALID <= 0;
    end
end

// =================================================================
// AXI4-Lite slave — write path (CLEAR register at 0x28)
// =================================================================
always @(posedge aclk) begin
    if (!aresetn) begin
        S_AXI_LITE_AWREADY <= 0;
        S_AXI_LITE_WREADY  <= 0;
        S_AXI_LITE_BVALID  <= 0;
        S_AXI_LITE_BRESP   <= 0;
    end else begin
        S_AXI_LITE_AWREADY <= 0;
        S_AXI_LITE_WREADY  <= 0;

        if (!S_AXI_LITE_BVALID) begin
            if (S_AXI_LITE_AWVALID && S_AXI_LITE_WVALID) begin
                S_AXI_LITE_AWREADY <= 1;
                S_AXI_LITE_WREADY  <= 1;
                S_AXI_LITE_BRESP   <= 2'b00;
                S_AXI_LITE_BVALID  <= 1;
                // address 0x28 → bits[5:2] = 4'hA
                // do_clear is a wire — no assignment needed here
            end
        end

        if (S_AXI_LITE_BVALID && S_AXI_LITE_BREADY)
            S_AXI_LITE_BVALID <= 0;
    end
end

endmodule
