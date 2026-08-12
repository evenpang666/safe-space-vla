#!/usr/bin/env python3
"""Export a binary XYZRGB PLY file as an offline, interactive HTML viewer.

The generated page has no CDN or Python dependency: open it directly in a
modern browser, then drag to rotate and use the mouse wheel to zoom.
"""

from __future__ import annotations

import argparse
import base64
import struct
from pathlib import Path


VERTEX_STRUCT = struct.Struct("<fffBBB")


def load_xyzrgb_binary_ply(path: Path) -> list[tuple[float, float, float, int, int, int]]:
    """Load the XYZRGB binary-little-endian PLY written by this project."""
    with path.open("rb") as stream:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            byte = stream.read(1)
            if not byte:
                raise ValueError("PLY header is incomplete")
            header.extend(byte)
        header_text = header.decode("ascii")
        if "format binary_little_endian 1.0" not in header_text:
            raise ValueError("only binary_little_endian PLY is supported")
        vertex_line = next(
            (line for line in header_text.splitlines() if line.startswith("element vertex ")),
            None,
        )
        if vertex_line is None:
            raise ValueError("PLY does not declare vertices")
        vertex_count = int(vertex_line.split()[-1])
        data = stream.read(vertex_count * VERTEX_STRUCT.size)
    if len(data) != vertex_count * VERTEX_STRUCT.size:
        raise ValueError("PLY vertex data is incomplete")
    return [
        VERTEX_STRUCT.unpack_from(data, offset)
        for offset in range(0, len(data), VERTEX_STRUCT.size)
    ]


def sample_vertices(
    vertices: list[tuple[float, float, float, int, int, int]], max_points: int
) -> list[tuple[float, float, float, int, int, int]]:
    if max_points <= 0 or len(vertices) <= max_points:
        return vertices
    step = len(vertices) / max_points
    return [vertices[int(index * step)] for index in range(max_points)]


def encode_vertices(vertices: list[tuple[float, float, float, int, int, int]]) -> str:
    packed = bytearray(len(vertices) * VERTEX_STRUCT.size)
    for index, vertex in enumerate(vertices):
        VERTEX_STRUCT.pack_into(packed, index * VERTEX_STRUCT.size, *vertex)
    return base64.b64encode(packed).decode("ascii")


HTML = """<!doctype html>
<html lang=\"zh-CN\">
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>{title}</title>
<style>
  body {{ margin:0; background:#10141b; color:#eaf0f8; font:14px system-ui,sans-serif; }}
  #panel {{ position:fixed; top:16px; left:16px; z-index:2; max-width:430px;
             padding:12px 14px; border-radius:9px; background:#1d2532e8; line-height:1.45; }}
  #panel h1 {{ margin:0 0 5px; font-size:16px; }}
  #panel p {{ margin:3px 0; color:#c6d2df; }}
  #count {{ color:#8fdbb0; }} canvas {{ width:100vw; height:100vh; display:block; cursor:grab; }}
  canvas:active {{ cursor:grabbing; }}
</style>
<div id=\"panel\"><h1>{title}</h1><p id=\"count\"></p><p>拖拽旋转；滚轮缩放；双击恢复初始视角。</p><p>{note}</p></div>
<canvas id=\"view\"></canvas>
<script>
const raw = atob('{data}');
const bytes = new Uint8Array(raw.length); for (let i=0; i<raw.length; ++i) bytes[i]=raw.charCodeAt(i);
const dv = new DataView(bytes.buffer), stride=15, n=bytes.length/stride;
const points=[];
let cx=0,cy=0,cz=0;
for(let i=0;i<n;i++){{const o=i*stride,x=dv.getFloat32(o,true),y=dv.getFloat32(o+4,true),z=dv.getFloat32(o+8,true);points.push([x,y,z,dv.getUint8(o+12),dv.getUint8(o+13),dv.getUint8(o+14)]);cx+=x;cy+=y;cz+=z;}}
cx/=n;cy/=n;cz/=n;
let radius=0; for(const p of points) radius=Math.max(radius,Math.hypot(p[0]-cx,p[1]-cy,p[2]-cz));
document.querySelector('#count').textContent=`交互预览：${{n.toLocaleString()}} / {source_count:,} 点（均匀抽样）`;
const canvas=document.querySelector('#view'),ctx=canvas.getContext('2d'); let yaw=-0.62,pitch=-0.18,dist=Math.max(radius*2.6,1),down=false,lastX=0,lastY=0;
function resize(){{const d=devicePixelRatio||1;canvas.width=innerWidth*d;canvas.height=innerHeight*d;ctx.setTransform(d,0,0,d,0,0);draw();}}
function draw(){{const w=innerWidth,h=innerHeight;ctx.fillStyle='#10141b';ctx.fillRect(0,0,w,h);const sy=Math.sin(yaw),cyw=Math.cos(yaw),sp=Math.sin(pitch),cp=Math.cos(pitch),f=Math.min(w,h)*0.85;for(const p of points){{let x=p[0]-cx,y=p[1]-cy,z=p[2]-cz;const x1=cyw*x-sy*z,z1=sy*x+cyw*z;const y1=cp*y-sp*z1,z2=sp*y+cp*z1;const q=dist-z2;if(q<=0.03)continue;const sx=w/2+f*x1/q,yy=h/2-f*y1/q;if(sx<-2||sx>w+2||yy<-2||yy>h+2)continue;const size=Math.max(0.55,Math.min(2.0,1.65*dist/q));ctx.fillStyle=`rgb(${{p[3]}},${{p[4]}},${{p[5]}})`;ctx.fillRect(sx,yy,size,size);}}}}
canvas.addEventListener('pointerdown',e=>{{down=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);}});
canvas.addEventListener('pointermove',e=>{{if(!down)return;yaw+=(e.clientX-lastX)*0.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-lastY)*0.008));lastX=e.clientX;lastY=e.clientY;draw();}});
canvas.addEventListener('pointerup',()=>down=false);
canvas.addEventListener('wheel',e=>{{e.preventDefault();dist=Math.max(radius*.25,Math.min(radius*10,dist*Math.exp(e.deltaY*.001)));draw();}},{{passive:false}});
canvas.addEventListener('dblclick',()=>{{yaw=-.62;pitch=-.18;dist=Math.max(radius*2.6,1);draw();}});
addEventListener('resize',resize); resize();
</script></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_ply", type=Path)
    parser.add_argument("output_html", type=Path)
    parser.add_argument("--max-points", type=int, default=80_000)
    parser.add_argument("--title", default="LingBot-Depth 单视角点云")
    parser.add_argument(
        "--note",
        default="使用参考内参构建，仅供观察；请使用真实 D435i 内参进行度量与融合。",
    )
    args = parser.parse_args()
    all_vertices = load_xyzrgb_binary_ply(args.input_ply)
    vertices = sample_vertices(all_vertices, args.max_points)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(
        HTML.format(
            title=args.title,
            note=args.note,
            data=encode_vertices(vertices),
            source_count=len(all_vertices),
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output_html} with {len(vertices):,}/{len(all_vertices):,} points")


if __name__ == "__main__":
    main()
