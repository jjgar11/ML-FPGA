# AXI DMA 7.1 — Registro cheatsheet (modo simple / direct)

Fuente: PG021 — Xilinx AXI DMA v7.1 Product Guide

## Dos canales independientes

| Canal | Dirección | Uso en este proyecto |
|-------|-----------|----------------------|
| **MM2S** — Memory-Map to Stream | DDR → IP | Manda los features de entrada al HLS IP |
| **S2MM** — Stream to Memory-Map | IP → DDR | Recibe los scores de salida del HLS IP |

Cada canal tiene su propio bloque de registros. Los offsets son relativos a la base UIO del DMA.

---

## MM2S — Canal de lectura (DDR → IP)

| Offset | Nombre   | R/W | Descripción |
|--------|----------|-----|-------------|
| `0x00` | MM2S_CR  | R/W | Control — bit 0: RS (Run=1/Stop=0) · bit 2: Reset (auto-clear, vacía FIFO) |
| `0x04` | MM2S_SR  | R/W | Status (ver tabla de bits abajo) |
| `0x18` | MM2S_SA  | W   | Source Address bits [31:0] — dirección física en DDR |
| `0x1C` | MM2S_SA_H| W   | Source Address bits [63:32] — para DDR > 4 GB |
| `0x28` | MM2S_LEN | W   | Número de bytes a leer. **Escribir aquí INICIA la transferencia.** |

---

## S2MM — Canal de escritura (IP → DDR)

| Offset | Nombre    | R/W | Descripción |
|--------|-----------|-----|-------------|
| `0x30` | S2MM_CR   | R/W | Control — mismo layout que MM2S_CR |
| `0x34` | S2MM_SR   | R/W | Status (ver tabla de bits abajo) |
| `0x48` | S2MM_DA   | W   | Destination Address bits [31:0] — dirección física en DDR |
| `0x4C` | S2MM_DA_H | W   | Destination Address bits [63:32] |
| `0x58` | S2MM_LEN  | R/W | Máximo de bytes a escribir. **Escribir aquí INICIA la transferencia.** Al terminar, leer este registro devuelve los bytes realmente recibidos (hasta TLAST). |

---

## Bits del registro de Control (CR)

```
bit 0  RS        Run/Stop — 1 = correr, 0 = parar (channel halted)
bit 2  Reset     Soft reset del canal (auto-clear). Vacía el SFIFO interno.
                 Esperar a que vuelva a 0 antes de continuar.
```

## Bits del registro de Status (SR)

La mayoría son W1C (Write-1-to-Clear): escribir un 1 en el bit lo limpia.

```
bit  0  Halted    Canal parado (RS=0, post-error, o post-transfer en algunos HW)
bit  1  Idle      Canal sin transferencia activa
bit  4  DMAIntErr Error interno — 2 causas posibles:
                    a) LEN alcanzado pero TLAST no llegó (IP mandó menos beats de lo esperado)
                    b) TLAST llegó antes de que se alcanzara LEN (IP mandó de más)
                    → Implica que el SFIFO interno puede quedar en estado corrupto.
                    → Para recuperar: hacer Reset (bit 2 de CR), lo que vacía el SFIFO.
bit  5  DMASlvErr Error de esclavo AXI (BRESP=SLVERR de la SmartConnect / DDR)
bit  6  DMADecErr Error de decodificación AXI (dirección inválida / fuera de rango)
bit 12  IOC_Irq   Transfer complete — el DMA terminó de transferir LEN bytes (W1C)
bit 13  Dly_Irq   Delay interrupt (no se usa en modo simple sin IRQ)
bit 14  Err_Irq   Qualquier error de los bits 4-6 generó interrupción (W1C)
```

---

## Secuencia de transferencia estándar

```
# Limpiar IOC del transfer anterior (W1C)
write CR_SR = SR_IOC_IRQ            # bit 12

# Poner canal en Run
write CR = 0x1                      # RS=1

# Configurar dirección y disparar
write DA / SA = dirección_física
write DA_H / SA_H = 0 (si < 4 GB)
write LEN = n_bytes                 # ← esto inicia la transferencia

# Esperar completación
while True:
    sr = read SR
    if sr & 0x70:  raise Error      # bits 4-6: algún error
    if sr & 0x1000: break           # bit 12: IOC → terminó
```

---

## Patrones de SR más comunes

| SR (hex)     | Significado |
|--------------|-------------|
| `0x00000000` | Canal corriendo (Running), sin IOC aún |
| `0x00001002` | Idle=1, IOC=1 — transferencia OK completada |
| `0x00000001` | Halted=1, RS=0 — canal parado normalmente |
| `0x00005011` | Halted+DMAIntErr+IOC+Err_Irq — LEN alcanzado pero llegaron más datos (fase 1 de nuestro test) |
| `0x00000011` | Halted+DMAIntErr sin IOC — SFIFO corrupto tras DMAIntErr previo, o TLAST llegó antes de LEN |
